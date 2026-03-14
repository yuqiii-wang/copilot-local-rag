const registerRankTool = require('./rankTool');
const registerCheckTool = require('./checkTool');
const registerReadMetadataTool = require('./readMetadataTool');
const registerReadContentTool = require('./readContentTool');
const registerNewCodeCheckTool = require('./newCodeCheckTool');
const registerReadRepoPromptsTool = require('./readRepoPromptsTool');
const registerCodeSplitterTool = require('./codeSplitterTool');

function createLanguageModelTools(deps) {
    const { vscode } = deps;

    function registerRepoAskLanguageModelTools() {
        if (!vscode.lm || typeof vscode.lm.registerTool !== 'function') {       
            return [];
        }

        const rankTool = registerRankTool(deps);
        const checkTool = registerCheckTool(deps);
        const readMetadataTool = registerReadMetadataTool(deps);
        const readContentTool = registerReadContentTool(deps);
        const newCodeCheckTool = registerNewCodeCheckTool(deps);
        const readRepoPromptsTool = registerReadRepoPromptsTool(deps);
        const codeSplitterTool = registerCodeSplitterTool(deps);

        return [rankTool, checkTool, readMetadataTool, readContentTool, newCodeCheckTool, readRepoPromptsTool, codeSplitterTool];
    }

    return {
        registerRepoAskLanguageModelTools
    };
}

module.exports = {
    createLanguageModelTools
};
