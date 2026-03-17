const registerRankTool = require('./rankTool');
const registerCheckTool = require('./checkTool');
const registerCodeDiffCheckTool = require('./codeCheckTool');
const registerReadRepoPromptsTool = require('./readRepoPromptsTool');
const registerCodeExploreTool = require('./codeExploreTool');

function createLanguageModelTools(deps) {
    const { vscode } = deps;

    function registerRepoAskLanguageModelTools() {
        if (!vscode.lm || typeof vscode.lm.registerTool !== 'function') {       
            return [];
        }

        const rankTool = registerRankTool(deps);
        const checkTool = registerCheckTool(deps);
        const codeDiffCheckTool = registerCodeDiffCheckTool(deps);
        const readRepoPromptsTool = registerReadRepoPromptsTool(deps);
        const codeExploreTool = registerCodeExploreTool(deps);

        return [rankTool, checkTool, codeDiffCheckTool, readRepoPromptsTool, codeExploreTool];
    }

    return {
        registerRepoAskLanguageModelTools
    };
}

module.exports = {
    createLanguageModelTools
};
