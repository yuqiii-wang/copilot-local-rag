const axios = require('axios');

class HttpManager {
    constructor() {
        this.queue = [];
        this.isProcessing = false;
        // Retry delays: 10s, 20s, 60s
        this.retryDelays = [10000, 20000, 60000];
        this.timeoutId = null;
    }

    request(config) {
        return new Promise((resolve, reject) => {
            this.queue.push({
                config,
                resolve,
                reject,
                failures: 0,
                readyAt: 0
            });
            this.processNext();
        });
    }

    async processNext() {
        if (this.isProcessing || this.queue.length === 0) return;

        const now = Date.now();
        
        // Find the first task that is ready to be processed
        const readyIndex = this.queue.findIndex(task => task.readyAt <= now);

        if (readyIndex === -1) {
            // No tasks ready yet, schedule the next check
            if (!this.timeoutId) {
                const earliestReadyAt = Math.min(...this.queue.map(t => t.readyAt));
                const waitTime = Math.max(0, earliestReadyAt - now);
                this.timeoutId = setTimeout(() => {
                    this.timeoutId = null;
                    this.processNext();
                }, waitTime);
            }
            return;
        }

        // A task is ready; clear any pending timeout
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }

        this.isProcessing = true;
        
        // Remove the ready task from the queue
        const task = this.queue.splice(readyIndex, 1)[0];

        try {
            // Apply infinite limits for large downloads/uploads
            const actualConfig = {
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
                ...task.config
            };
            
            const response = await axios(actualConfig);
            task.resolve(response.data);
        } catch (error) {
            task.failures += 1;
            console.error(`HTTP Request failed (${task.failures}/3) for ${task.config.url}: ${error.message}`);
            
            if (task.failures >= 3) {
                // If failed 3 times, remove completely by rejecting
                console.error(`Removing failed request after 3 attempts: ${task.config.url}`);
                task.reject(error);
            } else {
                // Cache failed http request, try after elongating interval
                const delay = this.retryDelays[task.failures - 1];
                task.readyAt = Date.now() + delay;
                this.queue.push(task);
            }
        }

        this.isProcessing = false;
        
        // Immediately try processing the next task in the queue
        this.processNext();
    }
}

// Singleton instance
const httpManager = new HttpManager();

/**
 * Shared function to construct auth headers for Jira and Confluence
 */
function getAuthHeaders(securityToken) {
    const headers = {};
    if (securityToken) {
        if (securityToken.startsWith('Bearer ') || securityToken.startsWith('Basic ')) {
            headers['Authorization'] = securityToken;
        } else if (securityToken.includes(':')) {
            headers['Authorization'] = `Basic ${Buffer.from(securityToken).toString('base64')}`;
        } else {
            headers['Authorization'] = `Bearer ${securityToken}`;
        }
    }
    return headers;
}

module.exports = {
    httpManager,
    getAuthHeaders
};
