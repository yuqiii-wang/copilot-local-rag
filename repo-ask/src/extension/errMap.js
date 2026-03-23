// errMap.js
// Centralized error mapping and message logic for sidebarController.js

/**
 * Maps error objects to user-friendly error messages and details for Confluence feedback submission.
 * @param {any} error - The error object thrown during feedback submission.
 * @returns {{ errorMessage: string, detailedError: string }}
 */
function mapFeedbackError(error) {
    let errorMessage = 'Failed to submit feedback. Please try again.';
    let detailedError = '';
    const body = error?.response?.data || error?.response?.body;
    let bodyMessage = '';
    if (typeof body === 'string') {
        bodyMessage = body.trim();
    } else if (body && typeof body === 'object') {
        bodyMessage = (body.message || body.error || JSON.stringify(body)).trim();
    }

    if (bodyMessage && bodyMessage !== '{}') {
        errorMessage = `Failed to connect to Confluence server: ${bodyMessage}`;
        detailedError = bodyMessage;
    } else if (error.message && error.message.includes('not configured')) {
        errorMessage = 'Confluence base URL not configured. Please set the repoAsk.confluence.url setting.';
        detailedError = error.message;
    } else if (error.code === 'ECONNABORTED' || (error.message && error.message.includes('timeout'))) {
        errorMessage = 'Failed to connect to Confluence server: Connection timed out. Please check your network connection and server URL.';
        detailedError = error.message;
    } else if (error.code === 'ECONNREFUSED') {
        errorMessage = 'Failed to connect to Confluence server: Connection refused. Please check if the server is running and the URL is correct.';
        detailedError = error.message || 'ECONNREFUSED';
    } else if (error.code) {
        errorMessage = `Failed to connect to Confluence server: ${error.code}. Please check your network connection and server URL.`;
        detailedError = error.message || error.code;
    } else if (error.response) {
        const status = error.response.status;
        if (status === 400) {
            errorMessage = 'Failed to connect to Confluence server: Bad Request (400). Please check your request data.';
        } else if (status === 401) {
            errorMessage = 'Failed to connect to Confluence server: Unauthorized (401). Please check your credentials.';
        } else if (status === 402) {
            errorMessage = 'Failed to connect to Confluence server: Payment Required (402).';
        } else if (status === 403) {
            errorMessage = 'Failed to connect to Confluence server: Forbidden (403). You do not have permission to perform this action.';
        } else if (status === 404) {
            errorMessage = 'Failed to connect to Confluence server: Page not found (404). Please check the URL and ensure the server is running.';
        } else if (status === 504) {
            errorMessage = 'Failed to connect to Confluence server: Gateway Timeout (504). The server took too long to respond.';
        } else if (status >= 500) {
            errorMessage = `Failed to connect to Confluence server: Server error (${status}). Please check if the server is running and accessible.`;
        } else {
            errorMessage = `Failed to connect to Confluence server: HTTP Error ${status}.`;
        }
        detailedError = error.response.statusText || '';
    } else if (error.message && error.message.includes('getaddrinfo')) {
        errorMessage = 'Failed to connect to Confluence server: Host not found. Please check the server URL.';
        detailedError = error.message;
    } else if (error.message) {
        errorMessage = `Failed to submit feedback: ${error.message}`;
        detailedError = error.message;
    }

    return { errorMessage, detailedError };
}

module.exports = {
    mapFeedbackError
};
