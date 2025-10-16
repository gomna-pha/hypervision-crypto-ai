/**
 * LIGHTWEIGHT LOADING & CACHING SYSTEM
 * Extracted from index.html for better performance
 */

// Progressive Loading Controller
class ProgressiveLoader {
    constructor() {
        this.currentStep = 0;
        this.steps = [
            { id: 'step-1', text: 'Loading core systems', progress: 20 },
            { id: 'step-2', text: 'Initializing connections', progress: 50 },
            { id: 'step-3', text: 'Loading dashboard', progress: 80 },
            { id: 'step-4', text: 'Platform ready', progress: 100 }
        ];
        this.loadingComplete = false;
    }

    updateProgress(stepIndex, customText = null) {
        if (stepIndex >= this.steps.length) return;
        
        const step = this.steps[stepIndex];
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const stepElement = document.getElementById(step.id);
        
        if (progressBar) {
            progressBar.style.width = step.progress + '%';
        }
        
        if (progressText) {
            progressText.textContent = customText || step.text;
        }
        
        if (stepElement) {
            stepElement.classList.remove('text-gray-400');
            stepElement.classList.add('text-green-600');
            stepElement.querySelector('div').classList.remove('border-gray-300');
            stepElement.querySelector('div').classList.add('border-green-500', 'bg-green-500');
        }
        
        this.currentStep = stepIndex;
    }

    async simulateLoading() {
        // Much faster loading - simplified steps
        await this.delay(200);
        this.updateProgress(0);
        
        await this.delay(300);
        this.updateProgress(1);
        
        await this.delay(400);
        this.updateProgress(2);
        
        await this.delay(300);
        this.updateProgress(3, 'Platform initialized successfully!');
        
        // Hide loading screen quickly
        await this.delay(400);
        this.hideLoadingScreen();
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        const app = document.getElementById('app');
        
        if (loadingScreen && app) {
            loadingScreen.style.opacity = '0';
            loadingScreen.style.transition = 'opacity 0.3s ease-out';
            
            setTimeout(() => {
                loadingScreen.style.display = 'none';
                app.style.display = 'block';
                app.style.opacity = '0';
                app.style.transition = 'opacity 0.3s ease-in';
                
                setTimeout(() => {
                    app.style.opacity = '1';
                    this.loadingComplete = true;
                }, 50);
            }, 300);
        }
    }
}

// Lightweight Client Cache
class ClientCache {
    constructor(maxSize = 50, ttl = 30000) { // Reduced size and TTL for speed
        this.cache = new Map();
        this.maxSize = maxSize;
        this.ttl = ttl;
    }

    set(key, value, customTTL = null) {
        if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        const entry = {
            value: value,
            timestamp: Date.now(),
            ttl: customTTL || this.ttl
        };

        this.cache.set(key, entry);
    }

    get(key) {
        const entry = this.cache.get(key);
        if (!entry) return null;

        if (Date.now() - entry.timestamp > entry.ttl) {
            this.cache.delete(key);
            return null;
        }

        return entry.value;
    }

    clear() {
        this.cache.clear();
    }
}

// Initialize systems
const progressiveLoader = new ProgressiveLoader();
const globalCache = new ClientCache();

// Cached fetch wrapper (simplified)
async function cachedFetch(url, options = {}, customTTL = null) {
    const cacheKey = `${url}_${JSON.stringify(options)}`;
    
    const cachedData = globalCache.get(cacheKey);
    if (cachedData) {
        return Promise.resolve(cachedData);
    }
    
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        globalCache.set(cacheKey, data, customTTL);
        return data;
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

// Export for global use
window.ProgressiveLoader = ProgressiveLoader;
window.ClientCache = ClientCache;
window.progressiveLoader = progressiveLoader;
window.globalCache = globalCache;
window.cachedFetch = cachedFetch;