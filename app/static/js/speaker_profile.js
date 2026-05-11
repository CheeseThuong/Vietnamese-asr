'use strict';
/**
 * @file: speaker_profile.js
 * @description: Nhận diện và phân biệt giọng nói (Speaker Diarization) thời gian thực trên trình duyệt sử dụng WebAudio API. Trích xuất đặc trưng MEL-bin và tính độ tương đồng Cosine.
 * @author: Nguyễn Trí Thượng
 * @project: VietASR Pro
 * @email: nguyentrithuong471@gmail.com
 * @github: CheeseThuong
 * @version: 2.0.0
 */

/**
 * @class SpeakerProfiler
 * @description Lớp xử lý âm thanh thời gian thực để phân loại người nói (Real-time speaker identification).
 * Hỗ trợ 2 chế độ:
 * 1. Chủ động đăng ký giọng (Register mode): Thu âm 3s mỗi người để lấy mẫu.
 * 2. Tự động phân cụm (Auto-clustering mode): Tự động tách giọng dựa trên sự thay đổi cao độ/năng lượng.
 */
class SpeakerProfiler {
    constructor(maxSpeakers = 4) {
        this.maxSpeakers = maxSpeakers;
        this.profiles = [];
        this.currentSpeaker = null;
        this.audioContext = null;
        this.analyser = null;
        this.autoMode = false;
        this._autoClusters = [];

        this.SPEAKERS = [
            { id: 'SPEAKER_00', label: 'Người nói 1', color: '#1565C0' },
            { id: 'SPEAKER_01', label: 'Người nói 2', color: '#B71C1C' },
            { id: 'SPEAKER_02', label: 'Người nói 3', color: '#1B5E20' },
            { id: 'SPEAKER_03', label: 'Người nói 4', color: '#E65100' },
            { id: 'UNKNOWN',    label: 'Không rõ',    color: '#7B1FA2' },
        ];
    }

    /** 
     * @function initAudioContext
     * @description Khởi tạo AudioContext và AnalyserNode từ luồng microphone 
     * @param {MediaStream} stream Luồng âm thanh từ microphone
     */
    initAudioContext(stream) {
        if (this.audioContext) return;
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 2048;
        this.analyser.smoothingTimeConstant = 0.5;
        const src = this.audioContext.createMediaStreamSource(stream);
        src.connect(this.analyser);
    }

    /** 
     * @function extractFeatures
     * @description Trích xuất các đặc trưng âm thanh: Năng lượng (Energy), Tỷ lệ qua điểm 0 (ZCR), Trọng tâm phổ (Spectral Centroid) và 13 vùng Mel (Mel-bins)
     * @returns {Object|null} Trả về đối tượng chứa các đặc trưng hoặc null nếu là khoảng lặng
     */
    extractFeatures() {
        if (!this.analyser) return null;

        const buf = new Float32Array(this.analyser.fftSize);
        this.analyser.getFloatTimeDomainData(buf);

        // RMS Energy
        const energy = Math.sqrt(buf.reduce((s, x) => s + x * x, 0) / buf.length);
        if (energy < 0.003) return null; // silence

        // Zero Crossing Rate
        let zcr = 0;
        for (let i = 1; i < buf.length; i++) {
            if ((buf[i] >= 0) !== (buf[i - 1] >= 0)) zcr++;
        }
        zcr /= buf.length;

        // Frequency data
        const freqBuf = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(freqBuf);

        // Spectral centroid
        let sumMag = 0, sumW = 0;
        freqBuf.forEach((m, i) => { sumMag += m; sumW += m * i; });
        const centroid = sumMag > 0 ? sumW / sumMag : 0;

        // 13 mel-spaced bins
        const mfcc = this._getMelBins(freqBuf, 13);

        return { energy, zcr, centroid, mfcc };
    }

    /** Get N mel-spaced frequency bins from FFT data */
    _getMelBins(freqBuf, n) {
        const sr = this.audioContext ? this.audioContext.sampleRate : 16000;
        const maxHz = sr / 2;
        const melMax = this._hzToMel(maxHz);
        const bins = [];
        for (let i = 0; i < n; i++) {
            const melLow = this._hzToMel(0) + (i / n) * melMax;
            const melHigh = this._hzToMel(0) + ((i + 1) / n) * melMax;
            const fLow = this._melToHz(melLow);
            const fHigh = this._melToHz(melHigh);
            const idxLow = Math.floor(fLow / maxHz * freqBuf.length);
            const idxHigh = Math.floor(fHigh / maxHz * freqBuf.length);
            let sum = 0, count = 0;
            for (let j = idxLow; j <= idxHigh && j < freqBuf.length; j++) { sum += freqBuf[j]; count++; }
            bins.push(count > 0 ? sum / count / 255 : 0);
        }
        return bins;
    }
    _hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
    _melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }

    /** Cosine similarity between two vectors */
    cosineSimilarity(a, b) {
        if (!a || !b || a.length !== b.length) return 0;
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
    }

    /** Build feature vector from Features object */
    _toVec(f) {
        return [...f.mfcc, f.zcr * 100, f.centroid / 100];
    }

    /**
     * @function registerSpeaker
     * @description Thu âm và trích xuất đặc trưng trong 1 khoảng thời gian (mặc định 3 giây) để tạo hồ sơ người nói (Profile).
     * @param {MediaStream} stream Luồng microphone
     * @param {number} durationMs Thời gian thu mẫu (milliseconds)
     * @returns {Promise<boolean>} True nếu thu mẫu thành công
     */
    async registerSpeaker(stream, durationMs = 3000) {
        this.initAudioContext(stream);
        const samples = [];
        return new Promise((resolve, reject) => {
            const interval = setInterval(() => {
                const f = this.extractFeatures();
                if (f) samples.push(this._toVec(f));
            }, 80);

            setTimeout(() => {
                clearInterval(interval);
                if (samples.length < 5) { reject(new Error('Không thu được đủ mẫu giọng')); return; }
                const avgVec = samples[0].map((_, i) => samples.reduce((s, v) => s + v[i], 0) / samples.length);
                const idx = this.profiles.length;
                this.profiles.push({ ...this.SPEAKERS[idx % 5], features: avgVec });
                resolve(true);
            }, durationMs);
        });
    }

    /**
     * @function identifyCurrentSpeaker
     * @description Nhận diện người đang nói trong thời gian thực dựa trên mẫu âm thanh hiện hành.
     * Thuật toán so sánh Cosine Similarity với các Profile đã lưu hoặc tự động gom cụm.
     * @returns {Object|null} Đối tượng người nói hoặc null (nếu là khoảng lặng)
     */
    identifyCurrentSpeaker() {
        if (!this.analyser) return null;
        const f = this.extractFeatures();
        if (!f) return null; // silence

        if (this.autoMode) return this._autoCluster(f);

        if (this.profiles.length === 0) return this.SPEAKERS[4]; // UNKNOWN

        const vec = this._toVec(f);
        let best = null, bestSim = -1;
        this.profiles.forEach(p => {
            if (!p.features) return;
            const sim = this.cosineSimilarity(vec, p.features);
            if (sim > bestSim) { bestSim = sim; best = p; }
        });

        if (bestSim < 0.60) return { ...this.SPEAKERS[4], confidence: Math.round(bestSim * 100) };
        return { ...best, confidence: Math.round(bestSim * 100) };
    }

    /** Auto clustering — no registration needed (pitch/energy based) */
    _autoCluster(f) {
        const vec = this._toVec(f);
        // Find nearest existing cluster center
        let best = null, bestSim = -1;
        this._autoClusters.forEach((c, i) => {
            const sim = this.cosineSimilarity(vec, c.center);
            if (sim > bestSim) { bestSim = sim; best = { ...this.SPEAKERS[i], idx: i }; }
        });
        if (bestSim > 0.75) {
            // Update cluster center (running average)
            const c = this._autoClusters[best.idx];
            c.center = c.center.map((v, i) => v * 0.95 + vec[i] * 0.05);
            return best;
        }
        // New cluster
        if (this._autoClusters.length < this.maxSpeakers) {
            const idx = this._autoClusters.length;
            this._autoClusters.push({ center: vec });
            return this.SPEAKERS[idx];
        }
        return this.SPEAKERS[4]; // UNKNOWN
    }

    /** Enable auto-clustering mode (skip registration) */
    enableAutoMode(stream) {
        this.autoMode = true;
        this._autoClusters = [];
        if (stream) this.initAudioContext(stream);
    }

    /** Reset all profiles */
    reset() {
        this.profiles = [];
        this._autoClusters = [];
        this.currentSpeaker = null;
        this.autoMode = false;
    }
}
