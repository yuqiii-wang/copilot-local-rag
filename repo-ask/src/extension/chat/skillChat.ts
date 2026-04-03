import fs from 'fs';
import path from 'path';
import { selectDefaultChatModel, withTimeout, LLM_RESPONSE_TIMEOUT_MS, emitThinking } from './shared';

const PROD_PLAN_ID = 'production-support-plan-skill';
const PROD_MAIN_ID = 'production-support-main-skill';

// Mirrors sanitizeFileSegment in utils.ts — used to predict the SKILL.md path.
function sanitizeSkillTitle(title: string): string {
    return String(title || 'item').toLowerCase().replace(/[^a-z0-9-_ ]+/g, '').trim().replace(/\s+/g, '-').slice(0, 64) || 'item';
}

/**
 * Ensures production-support-plan and production-support-main skill files exist in
 * <workspace>/.github/skills/. If either is absent from the docstore it is first
 * synced from the bundled default_docs, then written to .github/skills/ exactly
 * as if the user clicked "Add to Skills" in the sidebar.
 */
function ensureDefaultSkills(
    vscodeApi: any,
    deps: {
        documentService: any;
        readAllMetadata: () => any[];
        readDocumentContent: (id: string) => string | null;
        extensionPath: string;
    }
): void {
    const { documentService, readAllMetadata, readDocumentContent, extensionPath } = deps;
    const DEFAULT_IDS = [PROD_PLAN_ID, PROD_MAIN_ID];

    let allMeta = readAllMetadata();
    const missingFromStorage = DEFAULT_IDS.filter(id => !allMeta.find((m: any) => String(m.id) === id));
    if (missingFromStorage.length > 0) {
        documentService.syncDefaultDocs(extensionPath);
        allMeta = readAllMetadata();
    }

    const workspaceRoot = vscodeApi.workspace.workspaceFolders?.[0]?.uri?.fsPath;
    if (!workspaceRoot) return;

    for (const id of DEFAULT_IDS) {
        const meta = allMeta.find((m: any) => String(m.id) === id);
        if (!meta) continue;
        const safeTitle = sanitizeSkillTitle(meta.title || id);
        const skillFilePath = path.join(workspaceRoot, '.github', 'skills', safeTitle, 'SKILL.md');
        if (!fs.existsSync(skillFilePath)) {
            const content = readDocumentContent(id);
            if (content && content.trim().length > 0) {
                documentService.writeDocumentSkillFile(meta, content);
            }
        }
    }
}

/**
 * Skill command handler for @repoask /skill.
 *
 * Flow:
 *   1. Filter stored docs to type === 'skill'.
 *   2. Rank filtered skills against the user query; pick the single best match.
 *   3. Load the skill content and stream an LLM answer guided by the skill instructions.
 */
async function runSkillCommand(
    vscodeApi: any,
    prompt: string,
    response: any,
    deps: {
        documentService: any;
        readAllMetadata: () => any[];
        readDocumentContent: (id: string) => string | null;
        storagePath: string;
        extensionPath: string;
    },
    options: { request?: any } = {}
) {
    const { documentService, readAllMetadata, readDocumentContent, storagePath, extensionPath } = deps;

    if (!vscodeApi.lm || !vscodeApi.LanguageModelChatMessage) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    // Ensure production-support plan & main skills are present in .github/skills/
    ensureDefaultSkills(vscodeApi, { documentService, readAllMetadata, readDocumentContent, extensionPath });

    const PLAN_SKILL_ID = PROD_PLAN_ID;

    // ── 1. Collect all skill-type docs ───────────────────────────────────────
    const allMeta = readAllMetadata();
    const skillDocs = allMeta.filter((m: any) => String(m.type || '').toLowerCase() === 'skill');

    if (skillDocs.length === 0) {
        response.markdown(
            'No skill documents found in the docstore.\n\n' +
            'Add skill docs by setting `"type": "skill"` in their metadata, or sync Confluence docs tagged as skills.'
        );
        return;
    }

    // ── 2. Force production-support-plan as the starting skill ───────────────
    let topSkillMeta: any = skillDocs.find((m: any) => String(m.id) === PLAN_SKILL_ID);

    if (!topSkillMeta) {
        // Fallback: rank remaining skills if plan doc is not present
        emitThinking(response, `production-support-plan skill not found — searching ${skillDocs.length} skill(s) for the best match...`);
        const skillIdSet = new Set(skillDocs.map((m: any) => String(m.id)));
        const ranked = documentService.rankLocalDocuments(prompt, skillDocs.length * 3);
        const rankedSkills = (ranked || []).filter((d: any) => skillIdSet.has(String(d.id)));
        topSkillMeta = rankedSkills.length > 0
            ? skillDocs.find((m: any) => String(m.id) === String(rankedSkills[0].id))
            : skillDocs[0];
    } else {
        emitThinking(response, `Starting with production support plan skill...`);
    }

    // ── 3. Load skill content ────────────────────────────────────────────────
    const skillContent = readDocumentContent(String(topSkillMeta.id));
    if (!skillContent || skillContent.trim().length === 0) {
        response.markdown(
            `Skill **${topSkillMeta.title}** was matched but has no content.\n` +
            'Refresh it from the sidebar and try again.'
        );
        return;
    }

    // Emit a file reference so the Copilot UI shows the skill doc link
    if (storagePath && typeof response.reference === 'function') {
        const docPath = path.join(storagePath, String(topSkillMeta.id), 'content.md');
        response.reference(vscodeApi.Uri.file(docPath));
    }

    const skillTitle = topSkillMeta.title || String(topSkillMeta.id);
    const skillUrl = topSkillMeta.url || '';

    response.markdown(
        `**Skill selected:** ${skillUrl ? `[${skillTitle}](${skillUrl})` : `**${skillTitle}**`}\n\n---\n\n`
    );

    // ── 4. Run the skill: skill content as instructions + user query ─────────
    const vsModel = await selectDefaultChatModel(vscodeApi, options);
    if (!vsModel) {
        response.markdown('No language model is available in this VS Code session.');
        return;
    }

    emitThinking(response, `Running skill: ${skillTitle}...`);

    const fullPrompt = [
        'You are executing the following skill. Follow its instructions precisely to answer the user request.',
        '',
        '## Skill Instructions',
        skillContent,
        '',
        '---',
        '',
        '## User Request',
        prompt
    ].join('\n');

    try {
        const llmResponse = await withTimeout(
            vsModel.sendRequest(
                [vscodeApi.LanguageModelChatMessage.User(fullPrompt)],
                {},
                options?.request?.token
            ),
            LLM_RESPONSE_TIMEOUT_MS,
            null
        );

        if (!llmResponse) {
            response.markdown('Skill execution timed out. Please try again.');
            return;
        }

        if (llmResponse.stream) {
            for await (const chunk of llmResponse.stream) {
                if (chunk instanceof vscodeApi.LanguageModelTextPart) {
                    response.markdown(chunk.value);
                }
            }
        }
    } catch (err: any) {
        response.markdown(`Error running skill: ${err?.message || err}`);
    }
}

export { runSkillCommand };
