const fs = require('fs');
const path = require('path');

function collectMarkdownFilesRecursive(dirPath) {
    const files = [];
    if (!fs.existsSync(dirPath)) {
        return files;
    }

    for (const entry of fs.readdirSync(dirPath, { withFileTypes: true })) {
        const absolutePath = path.join(dirPath, entry.name);
        if (entry.isDirectory()) {
            files.push(...collectMarkdownFilesRecursive(absolutePath));
            continue;
        }

        if (entry.isFile() && entry.name.toLowerCase().endsWith('.md')) {
            files.push(absolutePath);
        }
    }

    return files;
}

function loadWorkspacePromptContext(vscode, maxChars = 0) {
    try {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return { text: '', fileCount: 0 };
        }

        const workspaceRoot = workspaceFolder.uri.fsPath;
        const promptsDirList = [
            path.join(workspaceRoot, '.github', '.prompts'),
        ];

        let markdownFiles = [];
        for (const dir of promptsDirList) {
            markdownFiles.push(...collectMarkdownFilesRecursive(dir));
        }
        
        markdownFiles = markdownFiles.sort((a, b) => a.localeCompare(b));

        if (markdownFiles.length === 0) {
            return { text: '', fileCount: 0 };
        }

        const sections = [];
        let usedFiles = 0;

        for (const filePath of markdownFiles) {
            const raw = fs.readFileSync(filePath, 'utf8');
            const content = String(raw || '').trim();
            if (!content) {
                continue;
            }

            const relativePath = path.relative(workspaceRoot, filePath).replace(/\\/g, '/');
            sections.push(`### ${relativePath}\n${content}`);
            usedFiles += 1;
        }

        const combined = sections.join('\n\n');
        if (!combined) {
            return { text: '', fileCount: 0 };
        }

        const limit = Number(maxChars);
        const hasLimit = Number.isFinite(limit) && limit > 0;

        return {
            text: hasLimit ? combined.slice(0, limit) : combined,
            fileCount: usedFiles
        };
    } catch {
        return { text: '', fileCount: 0 };
    }
}

module.exports = {
    loadWorkspacePromptContext
};
