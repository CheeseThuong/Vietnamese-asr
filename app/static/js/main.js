const BASE_URL = window.VIETASR_BASE_URL || '';
/**
 * @file: main.js
 * @description: File xử lý logic giao diện frontend, tương tác với trình duyệt (Web Speech API) và backend Flask (REST/XHR).
 * @author: Nguyễn Trí Thượng
 * @project: VietASR Pro
 * @email: nguyentrithuong471@gmail.com
 * @github: CheeseThuong
 * @version: 2.0.0
 */

'use strict';
// ===== UTILS =====
const $ = id => document.getElementById(id);
const $$ = s => document.querySelectorAll(s);

function supportsMediaDevices() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function supportsSpeechRecognition() {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

function cleanKeyword(keyword) {
    if (!keyword) return '';
    return keyword.replace(/[*#`_]/g, '').trim();
}

function formatAISummary(text) {
    if (!text) return '';
    
    // 1. Escape HTML to prevent XSS
    let escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
        
    // 2. Convert **text** to <strong>text</strong>
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 3. Convert `code` to <code>code</code>
    escaped = escaped.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // 4. Convert * text or - text to list items, and handle headings
    const lines = escaped.split('\n');
    let inList = false;
    const formattedLines = [];
    
    lines.forEach(line => {
        let trimmed = line.trim();
        
        // Handle headings starting with #
        if (trimmed.startsWith('#')) {
            if (inList) {
                formattedLines.push('</ul>');
                inList = false;
            }
            const headingText = trimmed.replace(/^#{1,6}\s*/, '').toUpperCase();
            if (headingText) {
                formattedLines.push(`<p><strong>${headingText}</strong></p>`);
            }
            return;
        }
        
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
            if (!inList) {
                formattedLines.push('<ul>');
                inList = true;
            }
            const itemText = trimmed.substring(2);
            formattedLines.push(`<li>${itemText}</li>`);
        } else {
            if (inList) {
                formattedLines.push('</ul>');
                inList = false;
            }
            if (trimmed) {
                formattedLines.push(`<p>${trimmed}</p>`);
            } else {
                formattedLines.push('<br>');
            }
        }
    });
    
    if (inList) {
        formattedLines.push('</ul>');
    }
    
    return formattedLines.join('\n');
}

const ICONS = {
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
    warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
};

/**
 * @function showToast
 * @description Hiển thị thông báo (toast message) góc màn hình
 * @param {string} msg Nội dung thông báo
 * @param {string} type Loại thông báo ('success', 'error', 'info', 'warning')
 * @param {number} dur Thời gian hiển thị (ms), mặc định 3500ms
 */
function showToast(msg, type='info', dur=3500) {
    const c=$('toastContainer'); if(!c) return;
    const t=document.createElement('div');
    t.className=`toast toast-${type}`;
    let icon = ICONS[type] || ICONS.info;
    if(type === 'success') icon = ICONS.check;
    t.innerHTML=`${icon}<span>${msg}</span>`;
    c.appendChild(t);
    setTimeout(()=>{t.style.opacity='0';setTimeout(()=>t.remove(),300);},dur);
}

function downloadFile(content,name,type='text/plain'){
    const b=new Blob([content],{type:type+';charset=utf-8'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(b); a.download=name; a.click();
    URL.revokeObjectURL(a.href);
}

function formatTime(s){
    s=Math.max(0,s||0);
    const m=Math.floor(s/60),sec=Math.floor(s%60);
    return `${m}:${String(sec).padStart(2,'0')}`;
}

function copyText(text){
    navigator.clipboard.writeText(text)
        .then(()=>showToast('Đã sao chép!','success'))
        .catch(()=>showToast('Lỗi sao chép','error'));
}

// ===== THEME =====
function initTheme(){
    const saved=localStorage.getItem('vietasr-theme')||'dark';
    document.documentElement.setAttribute('data-theme',saved);
    updateThemeIcon(saved);
}
function toggleTheme(){
    const c=document.documentElement.getAttribute('data-theme');
    const n=c==='dark'?'light':'dark';
    document.documentElement.setAttribute('data-theme',n);
    localStorage.setItem('vietasr-theme',n);
    updateThemeIcon(n);
}
function updateThemeIcon(t){
    const b=$('themeToggle'); if(!b) return;
    b.innerHTML=t==='dark'?'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>':'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>';
}

// ===== NAVIGATION =====
const PAGE_INFO={
    recording:{
        title:'Phòng Thu Âm',
        sub:'Chuyển đổi giọng nói thành văn bản trong thời gian thực'
    },
    upload:{
        title:'Tải File Âm Thanh',
        sub:'Tải lên và nhận dạng file âm thanh'
    },
    history:{
        title:'Lịch Sử',
        sub:'Xem lại các kết quả nhận dạng trước đó'
    },
    settings:{
        title:'Cài Đặt',
        sub:'Tuỳ chỉnh cấu hình hệ thống'
    }
};
function switchTab(tab){
    $$('.nav-item').forEach(n=>n.classList.toggle('active',n.dataset.tab===tab));
    $$('.section').forEach(s=>s.classList.remove('active'));
    const sec=$(tab+'Section'); if(sec) sec.classList.add('active');
    const info=PAGE_INFO[tab]||{};
    const pt=$('pageTitle'),ps=$('pageSubtitle');
    if(pt) pt.textContent=info.title||'';
    if(ps) ps.textContent=info.sub||'';
}

// ===== TABS =====
function initTabs(){
    $$('.tab-btn').forEach(btn=>btn.addEventListener('click',()=>{
        const p=btn.closest('.tabs-container'); if(!p) return;
        p.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
        p.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
        btn.classList.add('active');
        const t=$(btn.dataset.target); if(t) t.classList.add('active');
    }));
}

// ===== FETCH WITH TIMEOUT UTILITY =====
async function fetchWithTimeout(resource, options = {}) {
    const { timeout = 5000 } = options;
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const response = await fetch(resource, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
}

// ===== MODEL STATUS =====
async function checkModelStatus(){
    try{
        const r=await fetchWithTimeout(`${BASE_URL}/api/status`, { timeout: 5000 });
        const d=await r.json();
        const ind=$('modelStatusIndicator'),txt=$('modelStatusText'),
              dev=$('deviceInfoText'),gpu=$('gpuBadge');
        
        let label = 'Chưa tải';
        let status = 'idle';
        
        if (d.status === 'ready') {
            label = 'Sẵn sàng';
            status = 'ok';
        } else if (d.status === 'loading') {
            label = 'Đang tải...';
            status = 'warn';
        } else if (d.status === 'error') {
            label = 'Lỗi model';
            status = 'err';
        }
        
        // Sidebar indicators
        if (d.status === 'ready') {
            if(ind){ind.className='status-indicator status-online';}
            if(txt) txt.textContent='Sẵn sàng';
            if(dev) dev.textContent=d.device||'CPU';
            if(gpu){
                const span = gpu.querySelector('span');
                if (span) span.textContent=d.device||'CPU';
                if(d.device&&d.device.toLowerCase().includes('cuda')) gpu.classList.add('active-gpu');
            }
        } else if (d.status === 'loading') {
            if(ind){ind.className='status-indicator status-warning';}
            if(txt) txt.textContent='Đang tải...';
        } else {
            if(ind){ind.className='status-indicator status-offline';}
            if(txt) txt.textContent=d.status === 'error' ? 'Lỗi model' : 'Chưa tải';
        }
        
        // Main status card
        setScardStatus('scardModel', label, status, 'scardModelDot');
        
        // Handle Wav2Vec2 button state on the UI
        const wav2vecBtn = $('modeWav2VecBtn');
        if (wav2vecBtn) {
            const badge = wav2vecBtn.querySelector('.mode-pill-badge');
            if (d.status === 'loading') {
                if (badge) {
                    badge.textContent = 'Đang tải...';
                    badge.className = 'mode-pill-badge research warning';
                }
                // Show custom status if currently in wav2vec2 mode
                if (window._currentLiveMode === 'wav2vec2') {
                    const descText = $('modeDescText');
                    if (descText) descText.innerHTML = '🤖 <strong>Local Wav2Vec2 Mode</strong> — Mô hình đang được tải ở chế độ nền, vui lòng chờ...';
                }
            } else if (d.status === 'ready') {
                if (badge) {
                    badge.textContent = 'Nghiên cứu';
                    badge.className = 'mode-pill-badge research';
                }
                if (window._currentLiveMode === 'wav2vec2') {
                    const descText = $('modeDescText');
                    if (descText) descText.innerHTML = '🤖 <strong>Local Wav2Vec2 Mode</strong> — Chậm hơn khi chạy CPU. Dùng để minh họa mô hình nghiên cứu Wav2Vec2 fine-tuned.';
                }
            } else if (d.status === 'error') {
                if (badge) {
                    badge.textContent = 'Lỗi model';
                    badge.className = 'mode-pill-badge research error';
                }
            }
        }
    }catch(e){
        console.warn('Status check failed:',e);
        const ind=$('modelStatusIndicator'),txt=$('modelStatusText');
        if(ind) ind.className='status-indicator status-offline';
        if(txt) txt.textContent='Lỗi kết nối';
        setScardStatus('scardModel', 'Lỗi kết nối', 'err', 'scardModelDot');
    }
}

// ===== SUMMARY MODAL =====
function showSummaryModal(text){
    const m=$('summaryModal'),b=$('summaryModalBody');
    if(m) m.hidden=false;
    if(b) b.innerHTML=`<div style="white-space:pre-wrap;line-height:1.7">${text}</div>`;
}

/**
 * @function callSummarize
 * @description Gọi API tóm tắt, hiển thị kết quả vào modal
 * @param {string} text Nội dung transcript cần tóm tắt
 * @param {string} mode Chế độ tóm tắt
 * @param {object} sourceInfo Thông tin nguồn (dùng để lưu lịch sử)
 * @param {HTMLButtonElement|null} triggerBtn Nút đã bấm (sẽ bị disable trong khi xử lý)
 */
async function callSummarize(text, mode, sourceInfo, triggerBtn){
    if(!text||!text.trim()){showToast('Chưa có nội dung để tóm tắt','warning');return;}

    // Vô hiệu hóa nút để tránh spam double-click
    if(triggerBtn){
        if(triggerBtn.disabled) return; // Đang xử lý, bỏ qua
        triggerBtn.disabled = true;
        triggerBtn._origText = triggerBtn.innerHTML;
        triggerBtn.innerHTML = 'Đang xử lý...';
    }

    const m=$('summaryModal'),b=$('summaryModalBody');
    if(m) m.hidden=false;
    if(b) b.innerHTML='<div class="loading-placeholder"><svg class="shimmer-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg><br>Đang phân tích...</div>';
    try{
        const r=await fetch(`${BASE_URL}/api/summarize`,{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({text,mode:mode||'summary'})
        });
        const d=await r.json();

        if(d.success && d.summary) {
            // Hiển thị thông tin model và số chunk nếu có
            let metaHtml = '';
            if(d.model || d.chunks_count > 1){
                const modelLabel = d.model ? `🤖 ${d.model}` : '';
                const chunkLabel = d.chunks_count > 1 ? `📦 ${d.chunks_count} phần` : '';
                metaHtml = `<div style="font-size:0.78em;opacity:0.6;margin-bottom:10px">${[modelLabel,chunkLabel].filter(Boolean).join(' · ')}</div>`;
            }
            showSummaryModal(metaHtml + formatAISummary(d.summary));
            if (sourceInfo) {
                saveSummaryHistory({
                    summary: d.summary,
                    summaryMode: mode || 'summary',
                    source: sourceInfo
                });
            }
        } else if(d.summary_error === 'no_api_key') {
            if(b) b.innerHTML='<div style="color:var(--warning-color,#f59e0b)">&#9888; GEMINI_API_KEY chưa được cấu hình. Mở file <code>.env</code> và thêm API key.</div>';
        } else if(d.summary_error === 'rate_limit_429') {
            if(b) b.innerHTML='<div style="color:var(--warning-color,#f59e0b)">&#9203; Gemini API đang quá tải (429 Rate Limit). Vui lòng thử lại sau vài phút.</div>';
        } else if(d.summary_error === 'forbidden_403') {
            if(b) b.innerHTML='<div style="color:var(--error-color,#ef4444)">&#128683; API key bị từ chối quyền truy cập (403 Forbidden). Kiểm tra lại GEMINI_API_KEY trong file .env.</div>';
        } else if(d.summary_error === 'timeout') {
            if(b) b.innerHTML='<div style="color:var(--warning-color,#f59e0b)">&#8987; Gemini phản hồi quá chậm (Timeout). Thử lại sau ít phút.</div>';
        } else if(d.summary_error === 'gemini_client_unavailable') {
            if(b) b.innerHTML='<div style="color:var(--error-color,#ef4444)">&#9888; Thư viện google-genai chưa cài. Chạy: <code>pip install google-genai</code></div>';
        } else {
            if(b) b.innerHTML=`<div style="color:var(--error-color,#ef4444)">${d.error||'Lỗi tóm tắt không xác định'}</div>`;
        }
    }catch(e){
        if(b) b.innerHTML='<div style="color:var(--error-color,#ef4444)">Lỗi kết nối server</div>';
    } finally {
        // Khôi phục nút sau khi hoàn thành (dù thành công hay thất bại)
        if(triggerBtn){
            triggerBtn.disabled = false;
            if(triggerBtn._origText) triggerBtn.innerHTML = triggerBtn._origText;
        }
    }
}

// ===== VISUALIZER =====
let _vizAnim=null;
function initVisualizer(stream){
    const canvas=$('audioVisualizer'); if(!canvas) return;
    const ctx=canvas.getContext('2d');
    const ac=new (window.AudioContext||window.webkitAudioContext)();
    const an=ac.createAnalyser(); an.fftSize=256;
    ac.createMediaStreamSource(stream).connect(an);
    canvas.width=canvas.offsetWidth; canvas.height=canvas.offsetHeight;
    function draw(){
        _vizAnim=requestAnimationFrame(draw);
        const data=new Uint8Array(an.frequencyBinCount);
        an.getByteFrequencyData(data);
        ctx.clearRect(0,0,canvas.width,canvas.height);
        const bw=canvas.width/data.length*2.5;
        const dark=document.documentElement.getAttribute('data-theme')==='dark';
        data.forEach((v,i)=>{
            const h=(v/255)*canvas.height*0.85;
            const g=ctx.createLinearGradient(i*bw,canvas.height-h,i*bw,canvas.height);
            g.addColorStop(0,dark?'#818cf8':'#6366f1');
            g.addColorStop(1,dark?'#4f46e5':'#a5b4fc');
            ctx.fillStyle=g;
            ctx.fillRect(i*bw,canvas.height-h,bw-1,h);
        });
    }
    if(_vizAnim) cancelAnimationFrame(_vizAnim);
    draw();
}
function stopVisualizer(){if(_vizAnim){cancelAnimationFrame(_vizAnim);_vizAnim=null;}}

/**
 * @class VietASRRecorder
 * @description Lớp quản lý trạng thái ghi âm bằng Web Speech API
 * Quản lý vòng đời ghi âm: Bắt đầu, Tạm dừng, Tiếp tục, Dừng.
 */
class VietASRRecorder {
    constructor(){
        this.recognition=null;
        this.finalTranscript='';
        this.speakerLines=[];
        this.isRecording=false;
        this.isPaused=false;
        this.totalTimeMs=0;
        this.lastStartTime=null;
        this._timer=null;
        this._autoRestart=null;
        /* FIX: _isRestarting flag prevents double-start race condition when
           onend fires while a setTimeout restart is already queued */
        this._isRestarting=false;
        this.currentSpeaker=null;
        this.profiler=null;
        this._profilerInterval=null;
        this.mediaStream=null;
        /* FIX: Socket.IO connection for real-time text post-processing */
        this._socket=null;
        this._initSocket();
    }

    _initSocket(){
        /* FIX: Connect to Socket.IO only if the library is available */
        if(typeof io === 'undefined') return;
        try{
            this._socket = io({ transports: ['websocket','polling'] });
            this._socket.on('text_corrected', (data)=>{
                /* Replace the last speaker line with the corrected version */
                if(!data || !data.corrected || data.corrected===data.original) return;
                if(this.speakerLines.length===0) return;
                const last = this.speakerLines[this.speakerLines.length-1];
                /* Only update if original matches (no stale correction) */
                if(last.text.trim()===data.original.trim()){
                    last.text = data.corrected;
                    /* Also fix the full finalTranscript */
                    this.finalTranscript = this.speakerLines.map(l=>l.text).join(' ');
                    this._renderLines();
                }
            });
            this._socket.on('transcript_update', (data)=>{
                if (window._currentLiveMode !== 'wav2vec2') return;
                console.log("[ASR-Update] Interim/Final received:", data);
                
                const ir = $('interimResult');
                const labelEl = document.querySelector('#interimBox .interim-label');
                
                if (labelEl && data.status_message) {
                    labelEl.textContent = data.status_message;
                }
                
                if (data.is_final) {
                    if (ir) ir.textContent = '';
                    
                    if (data.text) {
                        const newSegment = data.text.trim();
                        const lastLine = this.speakerLines.length > 0 ? this.speakerLines[this.speakerLines.length - 1].text.trim() : '';
                        if (newSegment && newSegment.toLowerCase() !== lastLine.toLowerCase()) {
                            this.finalTranscript = data.full_text;
                            
                            // Word count speed guard check
                            const words = this.finalTranscript.trim().split(/\s+/).filter(w=>w).length;
                            const elapsedSec = (this.totalTimeMs + (this.lastStartTime ? Date.now() - this.lastStartTime : 0)) / 1000;
                            if (elapsedSec > 2 && words / elapsedSec > 15) {
                                console.warn('[Guard] Final transcript growth rate is abnormally high:', words / elapsedSec, 'words/sec');
                                showToast('Cảnh báo: Tốc độ nhận dạng bất thường. Vui lòng kiểm tra lại thiết bị thu âm.', 'warning');
                                this.stop();
                                return;
                            }
                            
                            this.speakerLines.push({
                                text: newSegment,
                                speaker: '',
                                label: '',
                                color: '',
                                time: elapsedSec
                            });
                            this._renderLines();
                        }
                    }
                    this._updateMeta();
                    
                } else {
                    if (data.text && data.text.trim()) {
                        if (ir) ir.textContent = data.text.trim();
                    } else if (ir) {
                        ir.textContent = '';
                    }
                }
                
                const latencyBadge = $('latencyBadge');
                if (latencyBadge && data.latency) {
                    latencyBadge.textContent = `${data.latency}ms`;
                    latencyBadge.hidden = false;
                }
            });
        } catch(e){
            console.warn('[VietASR-RT] Socket.IO init failed:', e);
        }
    }

    _initRecognition(){
        const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
        if(!SR) return false;
        this.recognition=new SR();
        this.recognition.lang=($('langSelect')||{}).value||'vi-VN';
        this.recognition.continuous=true;
        this.recognition.interimResults=true;
        this.recognition.maxAlternatives=1;

        this.recognition.onresult=(e)=>this._onResult(e);
        this.recognition.onend=()=>this._onEnd();
        this.recognition.onerror=(e)=>{
            console.warn('[VietASR-RT] SpeechRecognition error:', e.error);
            /* FIX: Also restart on network and audio-capture errors, not just no-speech */
            const recoverable=['no-speech','network','audio-capture','aborted'];
            if(recoverable.includes(e.error)){
                if(this.isRecording && !this.isPaused) this._scheduleRestart();
                return;
            }
            if(e.error==='not-allowed'){
                showToast('Microphone bị từ chối quyền truy cập. Vui lòng cấp quyền và thử lại.','error');
                this.stop();
            } else {
                console.error('[VietASR-RT] Unrecoverable error:', e.error);
            }
        };
        return true;
    }

    _onResult(e){
        let interim='', newFinal='';
        for(let i=e.resultIndex;i<e.results.length;i++){
            const t=e.results[i][0].transcript;
            if(e.results[i].isFinal) newFinal+=t+' ';
            else interim=t;
        }
        if(newFinal.trim()){
            this.finalTranscript+=newFinal;
            const spk=this.currentSpeaker;
            const segment={
                text:newFinal.trim(),
                speaker:spk?spk.id:'',
                label:spk?spk.label:'',
                color:spk?spk.color:'',
                time:(this.totalTimeMs + (this.lastStartTime ? Date.now() - this.lastStartTime : 0))/1000
            };
            this.speakerLines.push(segment);
            this._renderLines();
            /* FIX: Send final segment to backend for dictionary/Gemini correction */
            if(this._socket && this._socket.connected){
                this._socket.emit('correct_text', { text: segment.text });
            }
        }
        const ir=$('interimResult');
        if(ir) ir.textContent=interim;
        this._updateMeta();
    }

    _scheduleRestart(){
        /* FIX: Centralised restart with guard to prevent double-start */
        if(this._isRestarting) return;
        this._isRestarting=true;
        clearTimeout(this._autoRestart);
        this._autoRestart=setTimeout(()=>{
            this._isRestarting=false;
            if(!this.isRecording || this.isPaused) return;
            try{ this.recognition.start(); }
            catch(startErr){
                console.warn('[VietASR-RT] Restart failed, reinitialising:', startErr);
                this._initRecognition();
                try{ this.recognition.start(); } catch(e2){}
            }
        }, 350);
    }

    _onEnd(){
        if(this.isRecording&&!this.isPaused) this._scheduleRestart();
    }

    _renderLines(){
        const c=$('speakerLines'); if(!c) return;
        c.innerHTML='';
        this.speakerLines.forEach(l=>{
            const div=document.createElement('div');
            div.className='speaker-line';
            if(l.color) div.style.borderLeftColor=l.color;
            
            let meta = '';
            const timeStr = formatTime(l.time);
            
            if (l.label) {
                meta = `<div class="speaker-line-meta">
                    <span class="speaker-badge" style="background:${l.color}">${l.label}</span>
                    <span class="timestamp">${timeStr}</span>
                </div>`;
            } else {
                meta = `<div class="speaker-line-meta">
                    <span class="timestamp" style="margin-left: 0;">[${timeStr}]</span>
                </div>`;
            }
            
            div.innerHTML=meta+`<div class="speaker-line-text">${l.text}</div>`;
            c.appendChild(div);
        });
        const ta=$('transcriptArea');
        if(ta) ta.scrollTop=ta.scrollHeight;
    }

    _updateMeta(){
        const wc=$('wordCount');
        if(wc) wc.textContent=this.finalTranscript.trim().split(/\s+/).filter(w=>w).length;
    }

    /**
     * @function start
     * @description Bắt đầu ghi âm và kết nối Microphone
     */
    async start(){
        if(this.isRecording) return;
        
        const mode = window._currentLiveMode || 'browser';
        if (mode === 'wav2vec2') {
            const modelStatus = $('scardModelVal') ? $('scardModelVal').textContent : '';
            if (modelStatus !== 'Sẵn sàng') {
                showToast('Mô hình Wav2Vec2 chưa sẵn sàng. Vui lòng chờ model tải xong hoặc chọn Browser Live Mode.', 'warning');
                return;
            }
            await this.startLocalWav2VecMode();
        } else {
            await this.startBrowserLiveMode();
        }
    }

    async startBrowserLiveMode(){
        console.log("[ASR-State] Start Browser Live Mode");
        if(!this._initRecognition()){
            const bw=$('browserWarning');
            if(bw) {
                bw.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg> Trình duyệt này không hỗ trợ Browser Live Mode. Hãy dùng Microsoft Edge/Chrome hoặc chuyển sang Local Wav2Vec2 Mode. <br><small>Để có hỗ trợ microphone tốt nhất, hãy mở ứng dụng tại http://127.0.0.1:5000 bằng trình duyệt Chrome hoặc Microsoft Edge.</small>';
                bw.hidden=false;
            }
            return;
        }
        
        try{
            this.recognition.start();
            this.isRecording=true; this.isPaused=false;
            this._isRestarting=false;
            this.totalTimeMs=0;
            this.lastStartTime=Date.now();
            $('startRecordBtn').classList.add('recording');
            $('pauseRecordBtn').hidden=false;
            $('stopRecordBtn').disabled=false;
            const rs=$('recStatusText'); if(rs) rs.textContent='Đang ghi âm (Browser)...';
            this._startTimer();
            showToast('Bắt đầu ghi âm (Browser Live Mode)','success');
        }catch(e){
            console.error('[BrowserSpeech] Failed to start recognition:', e);
            showToast('Không thể khởi chạy nhận dạng giọng nói: '+e.message,'error');
            return;
        }

        // Tự động mở microphone cho visualizer & speaker detection (chỉ là tùy chọn)
        if (supportsMediaDevices()) {
            if (this.mediaStream) {
                initVisualizer(this.mediaStream);
                this._startSpeakerDetection();
            } else {
                try{
                    this.mediaStream=await navigator.mediaDevices.getUserMedia({audio:true});
                    initVisualizer(this.mediaStream);
                    this._startSpeakerDetection();
                }catch(e){
                    console.warn('[BrowserSpeech] Visualizer/Speaker detection optional features disabled:', e);
                    showToast('Tính năng sóng âm tắt (thiếu quyền microphone).','warning');
                }
            }
        } else {
            console.warn('[BrowserSpeech] navigator.mediaDevices.getUserMedia is not supported on this context/browser.');
        }
    }

    async startLocalWav2VecMode(){
        console.log("[ASR-State] Start Local Wav2Vec2 Mode");
        if (!supportsMediaDevices()) {
            showToast('Trình duyệt không hỗ trợ microphone API hoặc quyền mic bị chặn.','error');
            return;
        }
        
        // Initialize status label
        const labelEl = document.querySelector('#interimBox .interim-label');
        if (labelEl) labelEl.textContent = "Đang nhận dạng...";
        
        try{
            if (!this.mediaStream) {
                this.mediaStream=await navigator.mediaDevices.getUserMedia({audio:true});
            }
            
            if (!this._socket || !this._socket.connected) {
                this._initSocket();
            }
            
            this.isRecording=true; this.isPaused=false;
            this.totalTimeMs=0;
            this.lastStartTime=Date.now();
            $('startRecordBtn').classList.add('recording');
            $('pauseRecordBtn').hidden=false;
            $('stopRecordBtn').disabled=false;
            const rs=$('recStatusText'); if(rs) rs.textContent='Đang ghi âm (Wav2Vec2)...';
            this._startTimer();
            initVisualizer(this.mediaStream);
            this._startSpeakerDetection();
            
            // Web Audio API context at 16000Hz
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContextClass({ sampleRate: 16000 });
            this.audioSource = this.audioCtx.createMediaStreamSource(this.mediaStream);
            this.processorNode = this.audioCtx.createScriptProcessor(4096, 1, 1);
            
            this.audioSource.connect(this.processorNode);
            this.processorNode.connect(this.audioCtx.destination);
            
            this.processorNode.onaudioprocess = (e) => {
                if (!this.isRecording || this.isPaused) return;
                const inputData = e.inputBuffer.getChannelData(0);
                if (this._socket && this._socket.connected) {
                    this._socket.emit('audio_chunk', inputData.buffer);
                }
            };
            
            showToast('Bắt đầu ghi âm (Local Wav2Vec2)','success');
        }catch(e){
            console.error('[VietASR-RT] Wav2Vec2 start error:', e);
            if (e.name === 'NotAllowedError' || e.message.includes('Permission')) {
                showToast('Microphone đang bị chặn. Vui lòng bật quyền microphone cho 127.0.0.1:5000 hoặc localhost.','error');
            } else {
                showToast('Không thể mở microphone: '+e.message,'error');
            }
        }
    }

    /**
     * @function pause
     * @description Tạm dừng hoặc tiếp tục ghi âm
     * Nếu đang thu -> Ngắt kết nối tạm thời và chốt thời gian
     * Nếu đang dừng -> Khởi động lại API và cộng tiếp thời gian
     */
    pause(){
        if(!this.isRecording) return;
        this.isPaused=!this.isPaused;
        const btn=$('pauseRecordBtn');
        const mode = window._currentLiveMode || 'browser';
        
        if(this.isPaused){
            if (mode === 'browser') {
                try{this.recognition.stop();}catch(e){}
            }
            if(btn) btn.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
            $('startRecordBtn').classList.remove('recording');
            if(this.lastStartTime){
                this.totalTimeMs += Date.now() - this.lastStartTime;
                this.lastStartTime = null;
            }
            this._stopTimer();
            stopVisualizer();
            this._stopSpeakerDetection();
            const rs=$('recStatusText'); if(rs) rs.textContent='Tạm dừng';
        } else {
            if (mode === 'browser') {
                try{this.recognition.start();}catch(e){
                    this._initRecognition();
                    try{this.recognition.start();}catch(err){}
                }
            }
            if(btn) btn.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
            $('startRecordBtn').classList.add('recording');
            this.lastStartTime = Date.now();
            this._startTimer();
            initVisualizer(this.mediaStream);
            this._startSpeakerDetection();
            const rs=$('recStatusText'); if(rs) rs.textContent = mode === 'browser' ? 'Đang ghi âm (Browser)...' : 'Đang ghi âm (Wav2Vec2)...';
        }
    }

    stop(){
        console.log("[ASR-State] Stop Recording");
        this.isRecording=false; this.isPaused=false;
        const mode = window._currentLiveMode || 'browser';
        
        if (mode === 'browser') {
            /* FIX: Clear restart flag and timeout before stopping to prevent ghost restart */
            this._isRestarting=true;
            clearTimeout(this._autoRestart);
            try{this.recognition.stop();}catch(e){}
            this._isRestarting=false;
        } else {
            // Local Wav2Vec2 Mode: emit stop_recording event to the backend so it commits remaining audio
            console.log("[ASR-State] Emitting stop_recording to server.");
            if (this._socket && this._socket.connected) {
                this._socket.emit('stop_recording');
            }
            
            // Clean up nodes
            if (this.processorNode) {
                this.processorNode.disconnect();
                this.processorNode = null;
            }
            if (this.audioSource) {
                this.audioSource.disconnect();
                this.audioSource = null;
            }
            if (this.audioCtx) {
                this.audioCtx.close();
                this.audioCtx = null;
            }
        }
        
        if(this.lastStartTime){
            this.totalTimeMs += Date.now() - this.lastStartTime;
            this.lastStartTime = null;
        }
        this._stopTimer();
        this._stopSpeakerDetection();
        stopVisualizer();
        if(this.mediaStream){this.mediaStream.getTracks().forEach(t=>t.stop());this.mediaStream=null;}
        $('startRecordBtn').classList.remove('recording');
        $('pauseRecordBtn').hidden=true;
        const pBtn = $('pauseRecordBtn');
        if(pBtn) pBtn.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
        $('stopRecordBtn').disabled=true;
        
        // Reset interim Result to empty
        const ir=$('interimResult'); if(ir) ir.textContent='';
        const labelEl = document.querySelector('#interimBox .interim-label');
        if (labelEl) labelEl.textContent = "Đang nhận dạng...";
        const rs=$('recStatusText'); if(rs) rs.textContent='Sẵn sàng';
        showToast('Đã dừng ghi âm','info');
        if(this.finalTranscript.trim()) saveToHistory('recording',this.finalTranscript);
    }

    clear(){
        this.finalTranscript=''; this.speakerLines=[];
        this.totalTimeMs=0;
        this.lastStartTime=null;
        const sl=$('speakerLines'); if(sl) sl.innerHTML='';
        const ir=$('interimResult'); if(ir) ir.textContent='';
        const rt=$('recordingTimer'); if(rt) rt.textContent='0:00';
        this._updateMeta();
    }

    copy(){ copyText(this.finalTranscript); }

    exportTxt(){
        if(!this.finalTranscript.trim()){showToast('Chưa có nội dung','warning');return;}
        downloadFile(this.finalTranscript,'phong-thu-'+new Date().toISOString().slice(0,10)+'.txt');
    }

    async summarize(){
        // Tránh spam: nếu đang xử lý thì bỏ qua
        const btn = $('summarizeBtn');
        if(btn && btn.disabled) return;

        const mode=($('smartModeSelect')||{}).value||'summary';
        const elMs = this.totalTimeMs + (this.lastStartTime ? Date.now() - this.lastStartTime : 0);
        const sourceInfo = {
            type: 'recording',
            label: 'Phiên ghi âm ' + new Date().toLocaleString('vi-VN'),
            filename: null,
            duration: formatTime(elMs / 1000),
            wordCount: this.finalTranscript.trim().split(/\s+/).filter(w=>w).length,
            transcript: this.finalTranscript
        };
        // Truyền btn để tự động disable trong khi chờ
        await callSummarize(this.finalTranscript, mode, sourceInfo, btn);
    }

    _startTimer(){
        this._timer=setInterval(()=>{
            let current = this.totalTimeMs;
            if(this.lastStartTime) current += Date.now() - this.lastStartTime;
            const el = Math.floor(current / 1000);
            const rt=$('recordingTimer'); if(rt) rt.textContent=formatTime(el);
        },1000);
    }
    _stopTimer(){clearInterval(this._timer);this._timer=null;}

    _startSpeakerDetection(){
        if(!window.SpeakerProfiler||!this.profiler) return;
        this._profilerInterval=setInterval(()=>{
            try{
                const spk=this.profiler.identifyCurrentSpeaker();
                if(spk){this.currentSpeaker=spk;this._updateActivityDots(spk.id);}
            }catch(e){}
        },200);
    }
    _stopSpeakerDetection(){
        clearInterval(this._profilerInterval);
        this._profilerInterval=null;
        this.currentSpeaker=null;
    }
    _updateActivityDots(activeSpeakerId){
        $$('.activity-dot').forEach(d=>{
            d.classList.toggle('active',d.dataset.speaker===activeSpeakerId);
        });
    }
}

// Global recorder instance
let recorder = new VietASRRecorder();

// ===== UPLOAD =====
let lastUploadResult = null;

/**
 * @function handleFileUpload
 * @description Xử lý tải file âm thanh lên server thông qua XHR để theo dõi tiến trình
 * @param {File} file Đối tượng File cần tải lên
 */
async function handleFileUpload(file) {
    if(!file) return;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('ground_truth', ($('groundTruthInput')||{}).value||'');
    formData.append('enable_diarization', ($('enableDiarization')||{}).checked?'true':'false');
    formData.append('min_speakers', ($('minSpeakersSelect')||{}).value||'2');
    formData.append('max_speakers', ($('maxSpeakersSelect')||{}).value||'4');

    const prog=$('uploadProgress'), bar=$('uploadProgressBar'), ptxt=$('uploadProgressText');
    const panel=$('uploadResultPanel');
    if(prog) prog.hidden=false;
    if(panel) panel.hidden=true;
    if(bar) bar.style.width='10%';
    if(ptxt) ptxt.textContent=`Đang xử lý: ${file.name}`;

    // Use XHR for upload progress
    return new Promise((resolve)=>{
        const xhr = new XMLHttpRequest();
        xhr.upload.onprogress = (e)=>{
            if(e.lengthComputable && bar){
                const pct = Math.round(e.loaded/e.total*50);
                bar.style.width = pct+'%';
                if(ptxt) ptxt.textContent=`Đang tải lên... ${pct*2}%`;
            }
        };
        xhr.onload = ()=>{
            try{
                if(bar) bar.style.width='100%';
                const d = JSON.parse(xhr.responseText);
                setTimeout(()=>{if(prog) prog.hidden=true;}, 600);
                if(d.success){
                    d.filename = file.name;
                    lastUploadResult=d;
                    displayUploadResult(d);
                    showToast('Nhận dạng thành công!','success');
                    saveToHistory('upload', d.transcription, file.name);
                } else {
                    showToast(d.error||'Lỗi xử lý file','error');
                }
            } catch(e){ showToast('Lỗi phân tích kết quả','error'); }
            resolve();
        };
        xhr.onerror = ()=>{ showToast('Lỗi kết nối server','error'); if(prog) prog.hidden=true; resolve(); };
        xhr.open('POST',`${BASE_URL}/api/upload`);

        // Show audio preview
        const preview = document.getElementById('audioPreview');
        if (preview && file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
        }
        xhr.send(formData);
    });
}

function displayUploadResult(d){
    const panel=$('uploadResultPanel'); if(panel) panel.hidden=false;
    // Plain text — no character limit
    const fr=$('uploadFinalResult'); if(fr) fr.textContent=d.transcription||'';
    const wc=$('uploadWordCount'); if(wc) wc.textContent=d.word_count||0;
    const pt=$('uploadProcessTime'); if(pt) pt.textContent=(d.processing_time_seconds||0)+'s';
    const we=$('uploadWerText'); if(we) we.textContent=d.wer!=null?'WER: '+(d.wer*100).toFixed(1)+'%':'WER: —';
    const sc=$('uploadSpeakerCount'); if(sc) sc.textContent=d.speaker_count||'—';
    if(d.dialogue&&d.dialogue.length>0){
        renderDialogue(d.dialogue);
        renderSpeakerLegend('speakerLegend', d.dialogue);
        renderSpeakerStats(d.speaker_stats);
    }
}

function renderDialogue(lines){
    const c=$('dialogueContainer'); if(!c) return;
    c.innerHTML='';
    lines.forEach(l=>{
        const div=document.createElement('div');
        div.className='dialogue-line';
        div.style.borderLeftColor=l.color||'#666';
        div.dataset.speaker=l.speaker||'';
        div.innerHTML=`<span class="speaker-badge" style="background:${l.color||'#666'}">${l.label||'?'}</span>
            <span class="timestamp">${formatTime(l.start||0)}</span>
            <p class="dialogue-text">${l.text||''}</p>`;
        div.querySelector('.speaker-badge').addEventListener('click',()=>filterBySpeaker(l.speaker));
        c.appendChild(div);
    });
}

function renderSpeakerLegend(containerId, lines){
    const lg=$(containerId); if(!lg) return;
    const speakers={};
    lines.forEach(l=>{ if(l.speaker&&!speakers[l.speaker]) speakers[l.speaker]=l; });
    lg.innerHTML=''; lg.hidden=false;
    Object.values(speakers).forEach(s=>{
        const sp=document.createElement('span');
        sp.className='legend-item';
        sp.innerHTML=`<span class="legend-dot" style="background:${s.color}"></span>${s.label}`;
        sp.style.cursor='pointer';
        sp.addEventListener('click',()=>filterBySpeaker(s.speaker));
        lg.appendChild(sp);
    });
}

function renderSpeakerStats(stats){
    const g=$('speakerStatsGrid'); if(!g||!stats) return;
    g.innerHTML=''; g.hidden=false;
    Object.entries(stats).forEach(([,s])=>{
        const card=document.createElement('div');
        card.className='speaker-stat-card';
        card.style.borderLeftColor=s.color||'#666';
        card.innerHTML=`<strong>${s.label}</strong><br>
            <small>${s.words} từ | ${s.duration}s | ${s.turns} lượt | ${s.percentage}%</small>`;
        g.appendChild(card);
    });
}

let _activeFilter=null;
function filterBySpeaker(speaker){
    const lines=$$('.dialogue-line');
    if(_activeFilter===speaker){_activeFilter=null;lines.forEach(l=>l.classList.remove('dimmed','active'));return;}
    _activeFilter=speaker;
    lines.forEach(l=>{
        if(l.dataset.speaker===speaker){l.classList.remove('dimmed');l.classList.add('active');}
        else{l.classList.add('dimmed');l.classList.remove('active');}
    });
}

// Export functions
function exportUploadTxt(){if(!lastUploadResult) return; downloadFile(lastUploadResult.transcription||'','vietasr_transcript.txt');}
function exportUploadSrt(){
    if(!lastUploadResult) return;
    const lines=lastUploadResult.dialogue||[];
    if(!lines.length){downloadFile(lastUploadResult.transcription||'','vietasr_subtitle.srt');return;}
    let srt='';
    lines.forEach((l,i)=>{
        const ts=(s)=>{const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60),ms=Math.floor((s%1)*1000);return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')},${String(ms).padStart(3,'0')}`;};
        srt+=`${i+1}\n${ts(l.start||0)} --> ${ts(l.end||0)}\n[${l.label||'?'}] ${l.text||''}\n\n`;
    });
    downloadFile(srt,'vietasr_subtitle.srt');
}
function exportUploadJson(){if(!lastUploadResult) return; downloadFile(JSON.stringify(lastUploadResult,null,2),'vietasr_result.json','application/json');}

// ===== HISTORY =====
function saveToHistory(type,text,filename){
    if(!text||!text.trim()) return;
    const h=JSON.parse(localStorage.getItem('vietasr_history')||'[]');
    h.unshift({type,text:text.substring(0,500),filename,date:new Date().toLocaleString('vi-VN'),words:text.trim().split(/\s+/).filter(w=>w).length});
    if(h.length>50) h.length=50;
    localStorage.setItem('vietasr_history',JSON.stringify(h));
    renderHistory();
}
function renderHistory(){
    const list=$('historyList'); if(!list) return;
    const h=JSON.parse(localStorage.getItem('vietasr_history')||'[]');
    if(!h.length){list.innerHTML='<div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="48" height="48"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg><p>Chưa có lịch sử</p></div>';return;}
    list.innerHTML=h.map(item=>`<div class="history-card">
        <div class="history-card-header">
        <div class="history-type badge-${item.type}">${item.type==='upload'?'Tải lên':'Ghi âm'}</div>
            <span class="history-date">${item.date}</span>
        </div>
        <p class="history-text">${item.text}</p>
        <small>${item.words} từ${item.filename?' | '+item.filename:''}</small>
    </div>`).join('');
}

// ===== SUMMARY HISTORY =====
function saveSummaryHistory(data){
    const h=JSON.parse(localStorage.getItem('vietasr_summary_history')||'[]');
    const record = {
        id: 'sum_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
        createdAt: new Date().toISOString(),
        ...data
    };
    h.unshift(record);
    if(h.length>50) h.length=50;
    localStorage.setItem('vietasr_summary_history',JSON.stringify(h));
    renderSummaryHistory();
}

function renderSummaryHistory(){
    const list=$('summaryHistoryList'); if(!list) return;
    const h=JSON.parse(localStorage.getItem('vietasr_summary_history')||'[]');
    if(!h.length){
        list.innerHTML=`<div class="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72Z"></path></svg>
            <p>Chưa có lịch sử tóm tắt</p>
        </div>`;
        return;
    }
    
    list.innerHTML = h.map(item => {
        const timeStr = new Date(item.createdAt).toLocaleString('vi-VN');
        const modeMap = {
            'summary': 'Tóm tắt chung',
            'meeting': 'Biên bản cuộc họp',
            'notes': 'Ghi chú học tập',
            'translate': 'Dịch tiếng Anh'
        };
        const modeLabel = modeMap[item.summaryMode] || item.summaryMode;
        const srcTypeIcon = item.source.type === 'recording' ? '🎙 Ghi âm' : '📁 Tải file';
        
        return `
<div class="summary-card" data-id="${item.id}">
  <div class="summary-card-header">
    <div class="summary-card-meta">
      <span class="badge badge-ai">🤖 AI Assistant</span>
      <span class="summary-mode">${modeLabel}</span>
    </div>
    <span class="summary-time">${timeStr}</span>
  </div>

  <div class="summary-content">${formatAISummary(item.summary)}</div>

  <div class="summary-source">
    <div class="source-header">📌 Nguồn trích dẫn</div>
    <div class="source-info">
      <span class="source-type-badge ${item.source.type}">${srcTypeIcon}</span>
      <span class="source-label">${item.source.label}</span>
    </div>
    <div class="source-stats">
      <span>⏱ ${item.source.duration}</span>
      <span>📝 ${item.source.wordCount} từ</span>
    </div>
  </div>

  <div class="summary-transcript-toggle" onclick="toggleTranscript('${item.id}')">
    <span>Xem văn bản gốc</span>
    <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><polyline points="6 9 12 15 18 9"></polyline></svg>
  </div>
  <div class="summary-transcript-body" id="transcript-${item.id}" style="display:none">
    <div class="transcript-text">${item.source.transcript.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
  </div>

  <div class="summary-card-actions">
    <button onclick="copySummary('${item.id}')" title="Sao chép tóm tắt">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Sao chép
    </button>
    <button onclick="deleteSummaryHistory('${item.id}')" title="Xóa" class="btn-danger-ghost">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg> Xóa
    </button>
  </div>
</div>`;
    }).join('');
}

function deleteSummaryHistory(id) {
    if(!confirm('Xóa bản tóm tắt này? Hành động không thể hoàn tác.')) return;
    let h=JSON.parse(localStorage.getItem('vietasr_summary_history')||'[]');
    h = h.filter(x => x.id !== id);
    localStorage.setItem('vietasr_summary_history',JSON.stringify(h));
    renderSummaryHistory();
}

function toggleTranscript(id) {
    const el = $('transcript-' + id);
    if(!el) return;
    const toggleBtn = el.previousElementSibling;
    const chevron = toggleBtn.querySelector('.chevron');
    if(el.style.display === 'none') {
        el.style.display = 'block';
        if(chevron) chevron.style.transform = 'rotate(180deg)';
    } else {
        el.style.display = 'none';
        if(chevron) chevron.style.transform = 'rotate(0deg)';
    }
}

function copySummary(id) {
    const h=JSON.parse(localStorage.getItem('vietasr_summary_history')||'[]');
    const item = h.find(x => x.id === id);
    if(item) {
        copyText(item.summary);
    }
}


// ===== SPEAKER WIZARD =====
let wizardSpeakerCount=0;
function initSpeakerWizard(){
    $$('.wizard-count-btn').forEach(btn=>btn.onclick=()=>{
        $$('.wizard-count-btn').forEach(b=>b.classList.remove('selected'));
        btn.classList.add('selected');
        wizardSpeakerCount=parseInt(btn.dataset.count);
        buildWizardRegisterStep(wizardSpeakerCount);
        $('wizardStep0').hidden=true;
        $('wizardStepRegister').hidden=false;
    });
    const skip=$('skipWizardBtn');
    if(skip) skip.onclick=()=>{
        $('speakerWizard').hidden=true;
        if(recorder.profiler && recorder.mediaStream) recorder.profiler.enableAutoMode(recorder.mediaStream);
        recorder.start();
    };
    const start=$('startAfterSetup');
    if(start) start.onclick=()=>{
        $('speakerWizard').hidden=true;
        recorder.start();
    };
}

const SPK_COLORS=['#1565C0','#B71C1C','#1B5E20','#E65100'];
const SPK_LABELS=['Người nói 1','Người nói 2','Người nói 3','Người nói 4'];

function buildWizardRegisterStep(n){
    const c=$('wizardSpeakers'); if(!c) return;
    c.innerHTML='';
    for(let i=0;i<n;i++){
        const card=document.createElement('div');
        card.className='wizard-speaker-card';
        card.id=`wizCard${i}`;
        card.innerHTML=`<h4 style="color:${SPK_COLORS[i]}"><span class="legend-dot" style="background:${SPK_COLORS[i]};display:inline-block"></span> ${SPK_LABELS[i]}</h4>
            <p class="wizard-reg-prompt">Hãy đọc: <em>"Xin chào, tôi đang thử nghiệm hệ thống nhận dạng giọng nói"</em></p>
            <button class="btn btn-secondary" id="regBtn${i}" onclick="registerWizardSpeaker(${i})">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg> Đăng ký giọng 3s
            </button>
            <div class="wizard-progress" id="wizProg${i}"><div class="wizard-progress-bar" id="wizProgBar${i}" style="width:0%"></div></div>
            <div id="wizStatus${i}"></div>`;
        c.appendChild(card);
    }
    buildActivityDots(n);
}

function buildActivityDots(n){
    const dots=$('speakerDots'); if(!dots) return;
    dots.innerHTML='';
    for(let i=0;i<n;i++){
        const d=document.createElement('div');
        d.className='activity-dot';
        d.dataset.speaker=`SPEAKER_0${i}`;
        d.style.background=SPK_COLORS[i];
        d.title=SPK_LABELS[i];
        dots.appendChild(d);
    }
    $('speakerActivityBar').hidden=false;
}

async function registerWizardSpeaker(idx){
    if(!recorder.mediaStream){showToast('Cần microphone trước','error');return;}
    if(!window.SpeakerProfiler){showToast('SpeakerProfiler chưa sẵn sàng','error');return;}
    if(!recorder.profiler){recorder.profiler=new SpeakerProfiler(4);recorder.profiler.initAudioContext(recorder.mediaStream);}
    const btn=$(`regBtn${idx}`);
    const progBar=$(`wizProgBar${idx}`);
    const status=$(`wizStatus${idx}`);
    const card=$(`wizCard${idx}`);
    if(btn) btn.disabled=true;
    if(card) card.classList.add('recording-card');
    if(status) status.textContent='Đang nghe...';
    // Animate progress 0→100% over 3s
    let pct=0;
    const progInt=setInterval(()=>{pct+=3.5;if(progBar)progBar.style.width=Math.min(pct,100)+'%';},100);
    try{
        await recorder.profiler.registerSpeaker(recorder.mediaStream,3000);
        clearInterval(progInt);
        if(progBar) progBar.style.width='100%';
        if(status) status.innerHTML='<span class="reg-status-ok"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16" style="vertical-align:text-bottom"><polyline points="20 6 9 17 4 12"></polyline></svg> Đã đăng ký thành công</span>';
    } catch(e){
        clearInterval(progInt);
        if(status) status.textContent='⛔ Lỗi: '+e.message;
        if(btn) btn.disabled=false;
    }
    if(card) card.classList.remove('recording-card');
}

document.addEventListener('DOMContentLoaded',()=>{
    // Console debug capability logging
    console.log("[LiveMode]", window._currentLiveMode);
    console.log("[Support] SpeechRecognition:", supportsSpeechRecognition());
    console.log("[Support] mediaDevices:", supportsMediaDevices());
    console.log("[SecureContext]", window.isSecureContext);

    // Check browser support and security context context warning
    const bw=$('browserWarning');
    if(bw) {
        if(!supportsSpeechRecognition()){
            bw.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg> Trình duyệt này không hỗ trợ Browser Live Mode. Hãy dùng Microsoft Edge/Chrome hoặc chuyển sang Local Wav2Vec2 Mode. <br><small>Để có hỗ trợ microphone tốt nhất, hãy mở ứng dụng tại http://127.0.0.1:5000 bằng trình duyệt Chrome hoặc Microsoft Edge.</small>';
            bw.hidden=false;
        } else if(!window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1'){
            bw.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg> Cảnh báo: Ứng dụng chạy ngoài ngữ cảnh bảo mật (HTTPS/localhost). Microphone có thể bị chặn. <br><small>Để có hỗ trợ microphone tốt nhất, hãy mở ứng dụng tại http://127.0.0.1:5000 bằng trình duyệt Chrome hoặc Microsoft Edge.</small>';
            bw.hidden=false;
        } else {
            bw.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg> Để có hỗ trợ microphone tốt nhất, hãy mở ứng dụng tại <strong>http://127.0.0.1:5000</strong> bằng trình duyệt Chrome hoặc Microsoft Edge.';
            bw.hidden=false;
        }
    }

    initTheme();
    initTabs();
    initSpeakerWizard();
    checkModelStatus();
    renderHistory();
    renderSummaryHistory();

    // UPGRADE 2025: new init functions
    initStatusCards();
    initLiveModeSelector();
    initDemoMode();

    // Navigation
    $$('.nav-item').forEach(n=>n.onclick=()=>switchTab(n.dataset.tab));
    const tt=$('themeToggle'); if(tt) tt.onclick=toggleTheme;

    // Recording buttons
    const sr=$('startRecordBtn');
    if(sr) sr.onclick=async()=>{
        if(recorder.isRecording) return;
        setGeminiButtonState('recording');
        
        const mode = window._currentLiveMode || 'browser';
        console.log("[StartClick] Mode:", mode);
        
        if (mode === 'browser') {
            if(window.SpeakerProfiler && supportsMediaDevices()){
                try{
                    recorder.mediaStream=await navigator.mediaDevices.getUserMedia({audio:true});
                    $('speakerWizard').hidden=false;
                } catch(e){
                    console.warn('[StartButton] Optional media stream failed for speaker wizard:', e);
                    recorder.start();
                }
            } else {
                recorder.start();
            }
        } else {
            // Local Wav2Vec2 Mode: requires getUserMedia.
            if (!supportsMediaDevices()) {
                showToast("Trình duyệt không hỗ trợ microphone API hoặc quyền mic bị chặn.", "error");
                return;
            }
            if(window.SpeakerProfiler){
                try{
                    recorder.mediaStream=await navigator.mediaDevices.getUserMedia({audio:true});
                    $('speakerWizard').hidden=false;
                } catch(e){
                    console.error('[StartButton] Failed to get microphone for local mode:', e);
                    if (e.name === 'NotAllowedError' || e.message.includes('Permission')) {
                        showToast('Microphone đang bị chặn. Vui lòng bật quyền microphone cho 127.0.0.1:5000 hoặc localhost.','error');
                    } else {
                        showToast('Không thể mở microphone. Vui lòng kiểm tra quyền microphone trong trình duyệt và Windows Settings.', 'error');
                    }
                }
            } else {
                recorder.start();
            }
        }
    };
    const pr=$('pauseRecordBtn'); if(pr) pr.onclick=()=>recorder.pause();
    const st=$('stopRecordBtn');
    if(st) st.onclick=()=>{
        recorder.stop();
        setTimeout(()=>setGeminiButtonState('stopped'),200);
    };

    // Recording actions
    const cr=$('copyResultBtn'); if(cr) cr.onclick=()=>recorder.copy();
    const cl=$('clearResultBtn'); if(cl) cl.onclick=()=>{
        recorder.clear();
        // Hide summary area on clear
        const sa=$('recSummaryArea'); if(sa) sa.hidden=true;
        const kp=$('recKeywordsPanel'); if(kp) kp.hidden=true;
        setGeminiButtonState('empty');
    };
    const et=$('exportTxtBtn'); if(et) et.onclick=()=>recorder.exportTxt();

    // UPGRADE 2025: Gemini summary button (recording page) — inline display
    const sb=$('summarizeBtn');
    if(sb) sb.onclick=async()=>{
        if(sb.disabled) return;
        if(!recorder.finalTranscript.trim()){showToast('Chưa có nội dung','warning');return;}
        const mode=($('smartModeSelect')||{}).value||'summary';
        await callSummarizeInline(
            recorder.finalTranscript, mode, 'recording',
            'recSummaryArea','recSummaryBody','recSummaryModel',
            sb
        );
    };

    // UPGRADE 2025: Recording page — copy/download summary buttons
    const rcs=$('recCopySummaryBtn');
    if(rcs) rcs.onclick=()=>copyText(($('recSummaryBody')||{}).textContent||'');
    const rds=$('recDownloadSummaryBtn');
    if(rds) rds.onclick=()=>exportSummaryTxt('recording');

    // Upload
    const bf=$('browseFileBtn'),fi=$('fileInput'),dz=$('dropZone');
    if(bf) bf.onclick=()=>fi&&fi.click();
    if(fi) fi.onchange=(e)=>{if(e.target.files[0]) handleFileUpload(e.target.files[0]);};
    if(dz){
        dz.ondragover=(e)=>{e.preventDefault();dz.classList.add('drag-over');};
        dz.ondragleave=()=>dz.classList.remove('drag-over');
        dz.ondrop=(e)=>{e.preventDefault();dz.classList.remove('drag-over');if(e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);};
    }
    const ed=$('enableDiarization');
    if(ed) ed.onchange=()=>{const rc=$('speakerRangeControls');if(rc) rc.style.opacity=ed.checked?'1':'0.4';};

    // Upload actions
    const uc=$('uploadCopyBtn');
    if(uc) uc.onclick=()=>{
        const text = getUploadTranscriptText();
        copyText(text);
    };
    const uclr=$('uploadClearBtn'); if(uclr) uclr.onclick=()=>{
        if($('uploadFinalResult')) $('uploadFinalResult').textContent='';
        if($('uploadSegmentList')) $('uploadSegmentList').innerHTML='';
        if($('dialogueContainer')) $('dialogueContainer').innerHTML='';
        const ua=$('uploadSummaryArea'); if(ua) ua.hidden=true;
        const uk=$('uploadKeywordsPanel'); if(uk) uk.hidden=true;
        const uds=$('uploadDownloadSummaryBtn'); if(uds) uds.hidden=true;
        lastUploadResult=null;
    };
    const udl=$('uploadDownloadBtn'); if(udl) udl.onclick=exportUploadTxt;

    // UPGRADE 2025: Download summary (upload page)
    const uds=$('uploadDownloadSummaryBtn');
    if(uds) uds.onclick=()=>exportSummaryTxt('upload');

    // UPGRADE 2025: Summarize button (upload page) — inline display
    const usm=$('uploadSummarizeBtn');
    if(usm) usm.onclick=()=>{
        const text = getUploadTranscriptText();
        if(!text){showToast('Chưa có transcript','warning');return;}
        const mode=($('smartModeSelect')||{}).value||'summary';
        callSummarizeInline(
            text, mode, 'upload',
            'uploadSummaryArea','uploadSummaryBody','uploadSummaryModel',
            usm
        );
    };

    // Summary modal (kept for backward compat)
    const csm=$('closeSummaryModal'); if(csm) csm.onclick=()=>{const m=$('summaryModal');if(m)m.hidden=true;};
    const csb=$('closeSummaryBtn'); if(csb) csb.onclick=()=>{const m=$('summaryModal');if(m)m.hidden=true;};
    const cps=$('copySummaryBtn'); if(cps) cps.onclick=()=>copyText(($('summaryModalBody')||{}).textContent||'');

    // History
    const ch=$('clearHistoryBtn'); if(ch) ch.onclick=()=>{localStorage.removeItem('vietasr_history');renderHistory();showToast('Đã xóa lịch sử','info');};
    const csh=$('clearSummaryHistoryBtn'); if(csh) csh.onclick=()=>{localStorage.removeItem('vietasr_summary_history');renderSummaryHistory();showToast('Đã xóa lịch sử tóm tắt','info');};

    // Polling — check model status every 30s
    setInterval(checkModelStatus, 30000);
    // UPGRADE 2025: Poll config & gemini status once on load
    setTimeout(()=>{ loadServerConfig(); checkGeminiStatus(); }, 800);
});


// ===================== UPGRADE 2025: STATUS CARDS =====================

/**
 * Initialises status cards polling on load.
 * Fetches /api/config and /api/gemini-status once on DOMContentLoaded.
 */
function initStatusCards(){
    setScardStatus('scardServer', 'Đang kết nối...', 'idle', '');
    setScardStatus('scardGemini', 'Đang kiểm tra...', 'idle', 'scardGeminiDot');
    // Set live mode from localStorage preference
    const savedMode = localStorage.getItem('vietasr-live-mode') || 'browser';
    updateLiveModeScard(savedMode);
}

function setScardStatus(cardId, value, status, dotId){
    const card=$(cardId); if(!card) return;
    card.setAttribute('data-status', status||'idle');
    const val=card.querySelector('.scard-value'); if(val) val.textContent=value;
    if(dotId){
        const dot=$(dotId); if(!dot) return;
        dot.className='scard-dot';
        if(status==='ok') dot.classList.add('scard-dot-green');
        else if(status==='warn') dot.classList.add('scard-dot-warn');
        else if(status==='err') dot.classList.add('scard-dot-err');
        else dot.style.background='var(--color-text-3)';
    }
}

async function loadServerConfig(){
    try{
        const r=await fetchWithTimeout(`${BASE_URL}/api/config`, { timeout: 5000 });
        const d=await r.json();
        // Server card
        setScardStatus('scardServer', `v${d.server_version||'?'}`, 'ok', 'scardServerDot');
        // Device card
        const dev=d.device||'cpu';
        setScardStatus('scardDevice', dev.toUpperCase(), d.device&&d.device.toLowerCase().includes('cuda')?'ok':'idle', 'scardDeviceDot');
        if($('deviceInfoText')) $('deviceInfoText').textContent=dev.toUpperCase();
        if($('scardDeviceVal')) $('scardDeviceVal').textContent=dev.toUpperCase();
        if($('gpuBadgeText')) $('gpuBadgeText').textContent=dev.toUpperCase();
        // Default live mode from server
        const srvMode=d.default_live_mode||'browser';
        const userMode=localStorage.getItem('vietasr-live-mode')||srvMode;
        applyLiveMode(userMode, false);
        // Default live mode select in settings
        const dlms=$('defaultLiveModeSelect');
        if(dlms) dlms.value=userMode;
    }catch(e){
        console.error('Failed to load server config:', e);
        setScardStatus('scardServer','Offline / Lỗi','err','scardServerDot');
        setScardStatus('scardDevice','Không rõ','err','scardDeviceDot');
        setScardStatus('scardModel','Lỗi kết nối','err','scardModelDot');
    }
}

async function checkGeminiStatus(){
    try{
        const r=await fetchWithTimeout(`${BASE_URL}/api/gemini-status`, { timeout: 5000 });
        const d=await r.json();
        const statusMap={
            'ready':['Sẵn sàng','ok'],
            'disabled':['Disabled','idle'],
            'no_key':['Chưa cấu hình','warn'],
            'unavailable':['Không khả dụng','err'],
        };
        const [label,status]=statusMap[d.status]||['Không rõ','idle'];
        setScardStatus('scardGemini', label, status, 'scardGeminiDot');
        const gb=$('geminiStatusBadge');
        if(gb){
            gb.title=d.error||label;
            if(d.status==='ready') gb.style.opacity='1';
            else gb.style.opacity='0.5';
        }
    }catch(e){
        console.error('Failed to check Gemini/AI status:', e);
        setScardStatus('scardGemini','Lỗi kết nối','err','scardGeminiDot');
    }
}

function updateLiveModeScard(mode){
    const lv=mode==='browser'?'Browser Live':'Wav2Vec2';
    setScardStatus('scardLiveMode', lv, 'ok', 'scardLiveModeDot');
    if($('scardLiveModeVal')) $('scardLiveModeVal').textContent=lv;
}


// ===================== UPGRADE 2025: LIVE MODE SELECTOR =====================

const MODE_DESCRIPTIONS={
    browser:'🌐 <strong>Browser Live Mode</strong> — Nhận dạng nhanh, thời gian thực qua Web Speech API. Phù hợp demo trực tiếp.',
    wav2vec2:'🤖 <strong>Local Wav2Vec2 Mode</strong> — Chậm hơn khi chạy CPU, dùng để minh họa mô hình nghiên cứu.'
};

function initLiveModeSelector(){
    const browserBtn=$('modeBrowserBtn');
    const wav2vecBtn=$('modeWav2VecBtn');
    if(!browserBtn||!wav2vecBtn) return;

    // Restore saved preference
    const saved=localStorage.getItem('vietasr-live-mode')||'browser';
    applyLiveMode(saved, false);

    browserBtn.onclick=()=>applyLiveMode('browser', true);
    wav2vecBtn.onclick=()=>applyLiveMode('wav2vec2', true);

    // Settings page default live mode select
    const dlms=$('defaultLiveModeSelect');
    if(dlms){
        dlms.value=saved;
        dlms.onchange=()=>applyLiveMode(dlms.value, true);
    }
}

function applyLiveMode(mode, save){
    const browserBtn=$('modeBrowserBtn');
    const wav2vecBtn=$('modeWav2VecBtn');
    const descEl=$('modeDescText');
    const engineBadge=$('engineBadge');

    if(browserBtn) browserBtn.classList.toggle('active', mode==='browser');
    if(wav2vecBtn) wav2vecBtn.classList.toggle('active', mode==='wav2vec2');
    if(descEl) descEl.innerHTML=MODE_DESCRIPTIONS[mode]||MODE_DESCRIPTIONS.browser;
    if(engineBadge){
        engineBadge.textContent=mode==='browser'?'🌐 Browser · vi-VN':'🤖 Wav2Vec2 · Local';
    }
    updateLiveModeScard(mode);

    // Update settings select if it exists
    const dlms=$('defaultLiveModeSelect');
    if(dlms) dlms.value=mode;

    if(save) localStorage.setItem('vietasr-live-mode', mode);
    window._currentLiveMode=mode;
}


// ===================== UPGRADE 2025: GEMINI BUTTON STATE MACHINE =====================
/**
 * States: 'recording' (disabled), 'stopped' (enabled if transcript), 'empty' (disabled), 'processing' (disabled)
 */
function setGeminiButtonState(state){
    const btn=$('summarizeBtn'); if(!btn) return;
    if(state==='recording'){
        btn.disabled=true;
        btn.title='Chỉ dùng sau khi dừng ghi âm';
    } else if(state==='stopped'){
        const hasText=recorder&&recorder.finalTranscript&&recorder.finalTranscript.trim().length>0;
        btn.disabled=!hasText;
        btn.title=hasText?'Tóm tắt bằng AI Assistant':'Cần có transcript trước';
    } else if(state==='empty'){
        btn.disabled=true;
        btn.title='Cần có transcript trước';
    } else if(state==='processing'){
        btn.disabled=true;
        btn.title='Đang xử lý...';
    }
}


// ===================== UPGRADE 2025: INLINE SUMMARIZE =====================
/**
 * Gọi API tóm tắt và hiển thị kết quả inline trong panelAreaId.
 * Không dùng modal — hiển thị trực tiếp trong trang.
 * Gemini sẽ không bao giờ được gọi khi đang ghi âm.
 */
let _lastSummaryText = { recording: '', upload: '' };

async function callSummarizeInline(text, mode, context, areaId, bodyId, modelBadgeId, triggerBtn){
    if(!text||!text.trim()){showToast('Chưa có nội dung','warning');return;}

    // GUARD: Never call Gemini during recording
    if(recorder&&recorder.isRecording){
        showToast('Không thể tóm tắt khi đang ghi âm','warning');
        return;
    }

    if(triggerBtn){
        if(triggerBtn.disabled) return;
        triggerBtn.disabled=true;
        triggerBtn._origText=triggerBtn.innerHTML;
        triggerBtn.innerHTML='<svg class="shimmer-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg> Đang xử lý...';
    }

    const area=$(areaId); if(area){area.hidden=false;}
    const body=$(bodyId); if(body) body.innerHTML='<span style="opacity:0.5;font-style:italic">Đang phân tích với AI Assistant...</span>';
    const badge=$(modelBadgeId); if(badge) badge.textContent='';

    try{
        const r=await fetch(`${BASE_URL}/api/summarize`,{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({text,mode:mode||'summary'})
        });
        const d=await r.json();

        if(d.success&&d.summary){
            if(body){
                body.innerHTML=formatAISummary(d.summary);
            }
            if(badge&&d.model) badge.textContent=d.model;

            // Extract and render keywords from summary (simple extraction)
            _lastSummaryText[context]=d.summary;
            renderKeywordsFromText(d.summary, context==='recording'?'recKeywordsPanel':'uploadKeywordsPanel',
                                              context==='recording'?'recKeywordsList':'uploadKeywordsList');

            // Show download summary button (upload page)
            if(context==='upload'){
                const uds=$('uploadDownloadSummaryBtn'); if(uds) uds.hidden=false;
            }

            // Save to summary history
            saveSummaryHistory({
                summary: d.summary,
                summaryMode: mode||'summary',
                source:{
                    type:context,
                    label:(context==='recording'?'Phiên ghi âm ':'File ')+new Date().toLocaleString('vi-VN'),
                    filename:lastUploadResult?lastUploadResult.filename:null,
                    duration:context==='recording'?formatTime((recorder.totalTimeMs||0)/1000):'—',
                    wordCount:text.trim().split(/\s+/).filter(w=>w).length,
                    transcript:text
                }
            });
            showToast('Tóm tắt thành công','success');
        } else {
            const errMap={
                'no_api_key':'⚠ GEMINI_API_KEY chưa được cấu hình trong .env',
                'rate_limit_429':'⏳ AI Assistant API đang quá tải (429). Vui lòng thử lại sau.',
                'forbidden_403':'🚫 API key bị từ chối (403). Kiểm tra lại GEMINI_API_KEY trong file .env.',
                'timeout':'⌛ AI Assistant phản hồi quá chậm. Thử lại sau ít phút.',
                'gemini_client_unavailable':'⚠ Thư viện google-genai chưa cài. Chạy: pip install google-genai',
            };
            const errMsg=errMap[d.summary_error]||d.error||'Lỗi tóm tắt không xác định';
            if(body) body.innerHTML=`<span style="color:var(--color-warning)">${errMsg}</span>`;
            showToast('Tóm tắt thất bại','error');
        }
    }catch(e){
        if(body) body.innerHTML='<span style="color:var(--color-danger)">Lỗi kết nối server</span>';
        showToast('Lỗi kết nối','error');
    }finally{
        if(triggerBtn){
            triggerBtn.disabled=false;
            if(triggerBtn._origText) triggerBtn.innerHTML=triggerBtn._origText;
        }
    }
}


// ===================== UPGRADE 2025: SEGMENT DISPLAY =====================
/**
 * Renders segments with timestamps [HH:MM:SS - HH:MM:SS] text
 * Called from displayUploadResult when chunks array is available.
 */
function renderSegments(chunks){
    const list=$('uploadSegmentList');
    const plain=$('uploadPlainEditor');
    if(!list) return;

    if(!chunks||!chunks.length){
        if(list) list.innerHTML='';
        if(plain) plain.hidden=false;
        return;
    }

    if(plain) plain.hidden=true;
    list.innerHTML='';

    const tsLabel=(s)=>{
        s=Math.max(0,s||0);
        const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);
        return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
    };

    chunks.forEach((c,i)=>{
        const div=document.createElement('div');
        div.className='transcript-segment';
        const start=c.start!=null?c.start:(i*2);
        const end=c.end!=null?c.end:(start+2);
        const ts=start===0&&end===0?'':`[${tsLabel(start)} – ${tsLabel(end)}]`;
        const spkBadge=c.label?`<span class="speaker-badge" style="background:${c.color||'#666'};margin-right:6px">${c.label}</span>`:'';
        div.innerHTML=`<span class="segment-timestamp">${ts}</span><span class="segment-text">${spkBadge}${c.text||''}</span>`;
        list.appendChild(div);
    });
    // Auto scroll
    list.scrollTop=list.scrollHeight;
}


// ===================== UPGRADE 2025: KEYWORDS =====================
/**
 * Simple keyword extraction from summary text (top nouns, >4 chars)
 * For production, replace with backend NLP extraction.
 */
function renderKeywordsFromText(text, panelId, listId){
    const panel=$(panelId), list=$(listId);
    if(!panel||!list) return;
    // Clean markdown first
    const cleanText = text.replace(/[*#`_]/g, '');
    // Tách từ > 4 ký tự, unique, loại stopwords
    const stopwords=new Set(['không','được','những','trong','của','này','một','khi','và','các','có','cho','với','về','để','thì','là','đã','sẽ','đây','theo','như','từ','tại','nhưng','vì']);
    const words=[...new Set(
        cleanText.split(/\s+/)
            .map(w=>w.replace(/[.,!?;:()"""]/g,'').toLowerCase())
            .filter(w=>w.length>4&&!stopwords.has(w)&&/[a-zA-ZÀ-ỹ]/.test(w))
    )].slice(0,8);

    if(!words.length){panel.hidden=true;return;}
    panel.hidden=false;
    list.innerHTML=words.map(w=>`<span class="keyword-chip">${cleanKeyword(w)}</span>`).join('');
}


// ===================== UPGRADE 2025: EXPORT SUMMARY =====================
function exportSummaryTxt(context){
    const summaryText=_lastSummaryText[context]||'';
    const transcriptText = context==='recording'
        ? (recorder?recorder.finalTranscript:'')
        : getUploadTranscriptText();

    if(!summaryText&&!transcriptText){showToast('Không có nội dung để xuất','warning');return;}

    const now=new Date().toLocaleString('vi-VN');
    let content=`=== VietASR Pro — Kết Quả ===\nThời gian: ${now}\n\n`;
    if(transcriptText) content+=`--- TRANSCRIPT ---\n${transcriptText}\n\n`;
    if(summaryText) content+=`--- TÓM TẮT ---\n${summaryText}`;

    downloadFile(content,`vietasr_${context}_${Date.now()}.txt`);
}

function getUploadTranscriptText(){
    // Prefer segment list text if available
    const segs=$$('.segment-text');
    if(segs&&segs.length>0) return [...segs].map(s=>s.textContent).join('\n');
    return ($('uploadFinalResult')||{}).textContent||'';
}


// ===================== UPGRADE 2025: DISPLAY UPLOAD RESULT (OVERRIDE) =====================
/**
 * Override/extend displayUploadResult to also render segments and show upload model note.
 * Original function is still called for backward compat fields.
 */
const _origDisplayUploadResult=displayUploadResult;
displayUploadResult=function(d){
    // Call original for WER, word count, speaker stats, dialogue
    _origDisplayUploadResult(d);

    // UPGRADE 2025: Render segments with timestamps
    // Try chunks > dialogue > plain text fallback
    const chunks=(d.chunks&&d.chunks.length)?d.chunks
        :(d.dialogue&&d.dialogue.length)?d.dialogue
        :null;
    renderSegments(chunks);

    // Show plain text if no segments
    if(!chunks){
        const plain=$('uploadPlainEditor'); if(plain) plain.hidden=false;
        const fr=$('uploadFinalResult'); if(fr) fr.textContent=d.transcription||'';
    }

    // Reset summary area
    const ua=$('uploadSummaryArea'); if(ua) ua.hidden=true;
    const uk=$('uploadKeywordsPanel'); if(uk) uk.hidden=true;
    const uds=$('uploadDownloadSummaryBtn'); if(uds) uds.hidden=true;
};


// ===================== UPGRADE 2025: DEMO MODE =====================
function initDemoMode(){
    const check=$('demoModeCheck'); if(!check) return;
    // Restore saved demo mode
    const saved=localStorage.getItem('vietasr-demo-mode')==='true';
    check.checked=saved;
    applyDemoMode(saved);
    check.onchange=()=>{
        applyDemoMode(check.checked);
        localStorage.setItem('vietasr-demo-mode', check.checked);
    };
}

function applyDemoMode(on){
    if(on) document.body.classList.add('demo-mode');
    else document.body.classList.remove('demo-mode');
}