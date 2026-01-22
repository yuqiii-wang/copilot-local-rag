export interface RequestData {
    manualText: string;
    ocrText?: string;
    urls?: string[];
}

export class RequestManager {
    private static instance: RequestManager;
    private requests: Map<string, RequestData> = new Map();

    private constructor() {}

    public static getInstance(): RequestManager {
        if (!RequestManager.instance) {
            RequestManager.instance = new RequestManager();
        }
        return RequestManager.instance;
    }

    public createRequest(data: RequestData): string {
        const id = Math.random().toString(36).substring(2, 10);
        this.requests.set(id, data);
        return id;
    }

    public getRequest(id: string): RequestData | undefined {
        return this.requests.get(id);
    }

    public deleteRequest(id: string): void {
        this.requests.delete(id);
    }
}
