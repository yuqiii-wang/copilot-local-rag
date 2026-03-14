const confluenceApi = require('./confluenceApi');
const jiraApi = require('./jiraApi');

module.exports = {
    ...confluenceApi,
    ...jiraApi
};