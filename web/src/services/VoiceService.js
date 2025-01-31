export class VoiceService {
    constructor() {
        this.availableVoices = [];
        this.selectedVoices = new Set();
        this.currentVersion = null;
        this.availableVersions = [];
    }

    async loadVersions() {
        try {
            const response = await fetch('/v2/audio/versions');
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || 'Failed to load versions');
            }
            
            const data = await response.json();
            this.availableVersions = data.versions;
            this.currentVersion = data.current;
            return data;
        } catch (error) {
            console.error('Failed to load versions:', error);
            throw error;
        }
    }

    async loadVoices() {
        try {
            // Load versions first if not loaded
            if (!this.currentVersion) {
                await this.loadVersions();
            }

            const response = await fetch(`/v2/audio/voices?version=${this.currentVersion}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || 'Failed to load voices');
            }
            
            const data = await response.json();
            if (!data.voices?.length) {
                throw new Error('No voices available');
            }

            this.availableVoices = data.voices;
            
            // Select first voice if none selected
            if (this.selectedVoices.size === 0) {
                const firstVoice = this.availableVoices.find(voice => voice && voice.trim());
                if (firstVoice) {
                    this.addVoice(firstVoice);
                }
            }

            return {
                voices: this.availableVoices,
                version: data.version
            };
        } catch (error) {
            console.error('Failed to load voices:', error);
            throw error;
        }
    }

    getAvailableVoices() {
        return this.availableVoices;
    }

    getSelectedVoices() {
        return Array.from(this.selectedVoices);
    }

    getSelectedVoiceString() {
        return Array.from(this.selectedVoices).join('+');
    }

    async setVersion(version) {
        try {
            const response = await fetch('/v2/audio/version', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(version)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || 'Failed to set version');
            }
            
            const data = await response.json();
            this.currentVersion = data.current;
            this.availableVersions = data.versions;
            
            // Reload voices for new version
            await this.loadVoices();
            return data;
        } catch (error) {
            console.error('Failed to set version:', error);
            throw error;
        }
    }

    getCurrentVersion() {
        return this.currentVersion;
    }

    getAvailableVersions() {
        return this.availableVersions;
    }

    addVoice(voice) {
        if (this.availableVoices.includes(voice)) {
            this.selectedVoices.add(voice);
            return true;
        }
        return false;
    }

    removeVoice(voice) {
        return this.selectedVoices.delete(voice);
    }

    clearSelectedVoices() {
        this.selectedVoices.clear();
    }

    filterVoices(searchTerm) {
        if (!searchTerm) {
            return this.availableVoices;
        }
        
        const term = searchTerm.toLowerCase();
        return this.availableVoices.filter(voice => 
            voice.toLowerCase().includes(term)
        );
    }

    hasSelectedVoices() {
        return this.selectedVoices.size > 0;
    }
}

export default VoiceService;