const { registerOpenDetailsCommand } = require('./commands/openDetails');
const { registerRefreshAndParseCommands } = require('./commands/refreshParse');
const { registerCheckAndRankCommands } = require('./commands/searchRank');

function registerCoreCommands(deps) {
    return [
        ...registerOpenDetailsCommand(deps),
        ...registerRefreshAndParseCommands(deps),
        ...registerCheckAndRankCommands(deps)
    ];
}

module.exports = {
    registerCoreCommands
};
