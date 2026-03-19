/* ═══════════════════════════════════════════
   AI YouTube Lecture Summariser — Client Logic
   Tabs, Flashcards, Quiz, Chat, Analytics, PDF
   ═══════════════════════════════════════════ */

// ── State ─────────────────────────────────────
const state = {
    currentUrl: '',
    videoId: '',
    summaryMarkdown: '',
    flashcards: [],
    flashcardIndex: 0,
    quiz: [],
    quizIndex: 0,
    quizAnswers: [],
    quizScore: 0,
    quizTimer: null,
    quizSeconds: 0,
    chatHistory: [],
    analysis: null,
    loadedTabs: { summary: false, flashcards: false, quiz: false, chat: false, analytics: false },
};

// ── DOM Elements ──────────────────────────────
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Init ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // CRITICAL: Initialize core functionality individually so one missing DOM element doesn't crash the rest
    try { initFormSubmit(); } catch(e) { console.error("initFormSubmit failed", e); }
    try { initThemeToggle(); } catch(e) { console.error("initThemeToggle failed", e); }
    try { initTabs(); } catch(e) { console.error("initTabs failed", e); }
    try { initActions(); } catch(e) { console.error("initActions failed", e); }
    
    // UI flair securely isolated
    try { initCursor(); } catch(e) { console.error("Cursor failed:", e); }
    try { initScrollProgress(); } catch(e) { console.error("Scroll failed:", e); }
    try { initBentoGlow(); } catch(e) { console.error("Glow failed:", e); }
});

// ═══════════════════════════════════════════════
// CUSTOM CURSOR & SCROLL
// ═══════════════════════════════════════════════
function initCursor() {
    const dot = $('#cursorDot');
    const outline = $('#cursorOutline');
    
    // Prevent errors on touch devices
    if (!window.matchMedia("(pointer: fine)").matches) {
        dot.style.display = 'none';
        outline.style.display = 'none';
        return;
    }

    // Hide default cursor ONLY since we know JS is running
    document.body.style.cursor = 'none';

    let cursorX = 0, cursorY = 0;
    let outlineX = 0, outlineY = 0;

    window.addEventListener('mousemove', (e) => {
        cursorX = e.clientX;
        cursorY = e.clientY;
        dot.style.transform = `translate(${cursorX}px, ${cursorY}px)`;
    });

    // Smooth trailing effect for the outline
    function animateOutline() {
        outlineX += (cursorX - outlineX) * 0.15;
        outlineY += (cursorY - outlineY) * 0.15;
        outline.style.transform = `translate(${outlineX}px, ${outlineY}px)`;
        requestAnimationFrame(animateOutline);
    }
    animateOutline();

    // Add hover states to interactable elements
    const interactables = document.querySelectorAll('a, button, input, .feature-card, .flashcard, .quiz-option');
    interactables.forEach(el => {
        el.addEventListener('mouseenter', () => document.body.classList.add('cursor-hover'));
        el.addEventListener('mouseleave', () => document.body.classList.remove('cursor-hover'));
    });
}

function initScrollProgress() {
    const bar = $('#scrollProgress');
    window.addEventListener('scroll', () => {
        const scrollTop = window.scrollY;
        const docHeight = document.body.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        bar.style.width = scrollPercent + '%';
    });
}

function initBentoGlow() {
    const cards = document.querySelectorAll('.bento-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        });
    });
}

// ═══════════════════════════════════════════════
// THEME TOGGLE
// ═══════════════════════════════════════════════
function initThemeToggle() {
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);

    const toggleBtn = $('#themeToggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            updateThemeIcon(next);
        });
    }
}

function updateThemeIcon(theme) {
    const toggleBtn = $('#themeToggle');
    if (toggleBtn) {
        toggleBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
}

// ═══════════════════════════════════════════════
// FORM SUBMIT — Main URL input
// ═══════════════════════════════════════════════
function initFormSubmit() {
    $('#urlForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = $('#urlInput').value.trim();
        if (!url) return;

        state.currentUrl = url;
        resetState();
        showSkeleton(true);
        hideError();
        hideResults();

        try {
            const res = await fetch('/api/summarise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to summarise');

            state.videoId = data.video_id;
            state.summaryMarkdown = data.summary;
            state.loadedTabs.summary = true;

            showSkeleton(false);
            showResults();
            renderVideoEmbed(data.video_id);
            renderSummary(data.summary);
            switchTab('summary');
        } catch (err) {
            showSkeleton(false);
            let msg = err.message;
            if (msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('429') || msg.includes('daily limit')) {
                msg = '🔑 Your API key has run out of free usage for today. Please get a new API key from a different Google account at aistudio.google.com, update your .env file, and restart the server.';
            }
            showError(msg);
        }
    });
}

function resetState() {
    state.flashcards = [];
    state.flashcardIndex = 0;
    state.quiz = [];
    state.quizIndex = 0;
    state.quizAnswers = [];
    state.quizScore = 0;
    state.chatHistory = [];
    state.analysis = null;
    state.loadedTabs = { summary: false, flashcards: false, quiz: false, chat: false, analytics: false };
    if (state.quizTimer) clearInterval(state.quizTimer);
    state.quizSeconds = 0;
    // Reset chat
    const chatMsgs = $('#chatMessages');
    if (chatMsgs) chatMsgs.innerHTML = `
        <div class="chat-welcome">
            <div class="chat-welcome-icon">💬</div>
            <p>Ask any question about the lecture!</p>
        </div>`;
}

// ═══════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════
function initTabs() {
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = btn.dataset.tab;
            if (tab) switchTab(tab);
        });
    });
}

function switchTab(tabName) {
    console.log("Switching to tab:", tabName);
    
    // Update buttons
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');

    // Update content
    $$('.tab-content').forEach(c => c.classList.remove('active'));
    $(`#tab-${tabName}`).classList.add('active');

    // Lazy-load data for tabs
    if (!state.loadedTabs[tabName] && tabName !== 'summary') {
        loadTabData(tabName);
    }
}

async function loadTabData(tabName) {
    console.log("Loading data for:", tabName);
    const container = $(`#tab-${tabName}`);
    container.innerHTML = `<div class="tab-loading"><div class="spinner"></div><p>Generating ${tabName} data...</p></div>`;

    const endpoint = tabName === 'analytics' ? 'analyse' : tabName;

    try {
        const res = await fetch(`/api/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: state.currentUrl }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `Failed to load ${tabName}`);
        
        console.log("Data loaded for:", tabName);
        state.loadedTabs[tabName] = true;

        switch (tabName) {
            case 'flashcards':
                state.flashcards = data.flashcards || [];
                renderFlashcards();
                break;
            case 'quiz':
                state.quiz = data.quiz || [];
                renderQuiz();
                break;
            case 'chat':
                renderChatUI();
                break;
            case 'analytics':
                state.analysis = data.analysis;
                renderAnalytics();
                break;
        }
    } catch (err) {
        console.error("Tab load error:", err);
        container.innerHTML = `<div class="tab-loading"><p style="color:var(--accent-3);">❌ Could not generate ${tabName}: ${err.message}</p></div>`;
    }
}

// ═══════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════
function renderSummary(markdown) {
    const html = markdownToHtml(markdown);
    
    // Add text decoding animation wrapper
    $('#summaryContent').innerHTML = `<div class="decode-text">${html}</div>`;
}

function markdownToHtml(md) {
    let html = md
        // Code blocks
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Inline code
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // H3
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        // H2
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        // H1
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        // Unordered list
        .replace(/^\- (.*$)/gm, '<li>$1</li>')
        // Ordered list
        .replace(/^\d+\. (.*$)/gm, '<li>$1</li>')
        // Horizontal rule
        .replace(/^---$/gm, '<hr>')
        // Paragraphs - wrap standalone lines
        .replace(/^(?!<[hloupir])(.*\S.*)$/gm, '<p>$1</p>');

    // Wrap consecutive li tags in ul
    html = html.replace(/(<li>.*?<\/li>\n?)+/g, '<ul>$&</ul>');

    return html;
}

// ═══════════════════════════════════════════════
// FLASHCARDS
// ═══════════════════════════════════════════════
function renderFlashcards() {
    const container = $(`#tab-flashcards`);
    container.innerHTML = `
        <div class="flashcard-controls">
            <button class="fc-control-btn" id="fcPrev" title="Previous">◀</button>
            <span class="fc-counter" id="fcCounter">1 / ${state.flashcards.length}</span>
            <button class="fc-control-btn" id="fcNext" title="Next">▶</button>
            <button class="fc-control-btn" id="fcShuffle" title="Shuffle">🔀</button>
        </div>
        <div class="flashcard-container">
            <div class="flashcard" id="flashcard">
                <div class="flashcard-face flashcard-front">
                    <div class="flashcard-label">Question</div>
                    <div class="flashcard-text" id="fcQuestion"></div>
                    <div class="flashcard-hint">Click to flip</div>
                </div>
                <div class="flashcard-face flashcard-back">
                    <div class="flashcard-label">Answer</div>
                    <div class="flashcard-text" id="fcAnswer"></div>
                    <div class="flashcard-hint">Click to flip back</div>
                </div>
            </div>
        </div>
    `;

    state.flashcardIndex = 0;
    updateFlashcard();

    $('#flashcard').addEventListener('click', () => {
        $('#flashcard').classList.toggle('flipped');
    });

    $('#fcPrev').addEventListener('click', () => {
        state.flashcardIndex = (state.flashcardIndex - 1 + state.flashcards.length) % state.flashcards.length;
        flipReset();
        updateFlashcard();
    });

    $('#fcNext').addEventListener('click', () => {
        state.flashcardIndex = (state.flashcardIndex + 1) % state.flashcards.length;
        flipReset();
        updateFlashcard();
    });

    $('#fcShuffle').addEventListener('click', () => {
        state.flashcards = shuffleArray([...state.flashcards]);
        state.flashcardIndex = 0;
        flipReset();
        updateFlashcard();
    });
}

function updateFlashcard() {
    const card = state.flashcards[state.flashcardIndex];
    $('#fcQuestion').textContent = card.question;
    $('#fcAnswer').textContent = card.answer;
    $('#fcCounter').textContent = `${state.flashcardIndex + 1} / ${state.flashcards.length}`;
}

function flipReset() {
    const fc = $('#flashcard');
    if (fc) fc.classList.remove('flipped');
}

function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

// ═══════════════════════════════════════════════
// QUIZ
// ═══════════════════════════════════════════════
function renderQuiz() {
    state.quizIndex = 0;
    state.quizAnswers = new Array(state.quiz.length).fill(null);
    state.quizScore = 0;
    state.quizSeconds = 0;

    renderQuizQuestion();
    startQuizTimer();
}

function renderQuizQuestion() {
    const container = $(`#tab-quiz`);
    const q = state.quiz[state.quizIndex];
    const total = state.quiz.length;
    const progress = ((state.quizIndex) / total) * 100;
    const answered = state.quizAnswers[state.quizIndex];

    container.innerHTML = `
        <div class="quiz-header">
            <div class="quiz-progress">
                <span>Q${state.quizIndex + 1}/${total}</span>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width:${progress}%"></div>
                </div>
            </div>
            <div class="quiz-timer" id="quizTimer">${formatTime(state.quizSeconds)}</div>
        </div>
        <div class="quiz-question-card">
            <div class="quiz-question-num">Question ${state.quizIndex + 1}</div>
            <div class="quiz-question-text">${q.question}</div>
            <div class="quiz-options">
                ${q.options.map((opt, i) => {
                    let cls = '';
                    if (answered !== null) {
                        if (i === q.correctIndex) cls = 'correct disabled';
                        else if (i === answered && answered !== q.correctIndex) cls = 'wrong disabled';
                        else cls = 'disabled';
                    }
                    const letter = String.fromCharCode(65 + i);
                    return `<button class="quiz-option ${cls}" data-index="${i}" ${answered !== null ? 'disabled' : ''}>
                        <span class="option-letter">${letter}</span>
                        <span>${opt}</span>
                    </button>`;
                }).join('')}
            </div>
            <div class="quiz-explanation ${answered !== null ? 'show' : ''}" id="quizExplanation">
                💡 ${q.explanation}
            </div>
        </div>
        <div class="quiz-nav">
            ${state.quizIndex > 0 ? '<button class="quiz-nav-btn" id="quizPrev">← Previous</button>' : ''}
            ${state.quizIndex < total - 1
                ? '<button class="quiz-nav-btn primary" id="quizNext">Next →</button>'
                : (answered !== null ? '<button class="quiz-nav-btn primary" id="quizFinish">Finish Quiz 🎉</button>' : '')}
        </div>
    `;

    // Option click handlers
    if (answered === null) {
        container.querySelectorAll('.quiz-option').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.dataset.index);
                state.quizAnswers[state.quizIndex] = idx;
                if (idx === q.correctIndex) state.quizScore++;
                renderQuizQuestion();
            });
        });
    }

    // Nav handlers
    const prevBtn = $('#quizPrev');
    const nextBtn = $('#quizNext');
    const finishBtn = $('#quizFinish');

    if (prevBtn) prevBtn.addEventListener('click', () => { state.quizIndex--; renderQuizQuestion(); });
    if (nextBtn) nextBtn.addEventListener('click', () => { state.quizIndex++; renderQuizQuestion(); });
    if (finishBtn) finishBtn.addEventListener('click', showQuizScore);
}

function startQuizTimer() {
    if (state.quizTimer) clearInterval(state.quizTimer);
    state.quizTimer = setInterval(() => {
        state.quizSeconds++;
        const el = $('#quizTimer');
        if (el) el.textContent = formatTime(state.quizSeconds);
    }, 1000);
}

function formatTime(secs) {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}

function showQuizScore() {
    if (state.quizTimer) clearInterval(state.quizTimer);
    const total = state.quiz.length;
    const pct = Math.round((state.quizScore / total) * 100);
    let msg = '';
    
    // Confetti effect for good score!
    if (pct >= 70) {
        fireConfetti();
    }

    if (pct >= 90) msg = '🏆 Outstanding! You nailed it!';
    else if (pct >= 70) msg = '🌟 Great job! Solid understanding!';
    else if (pct >= 50) msg = '💪 Good effort! Keep studying!';
    else msg = '📚 Review the lecture and try again!';

    const container = $(`#tab-quiz`);
    container.innerHTML = `
        <div class="quiz-score">
            <div class="score-circle">
                <span>${pct}%</span>
                <span class="score-label">Score</span>
            </div>
            <div class="score-message">${msg}</div>
            <div class="score-detail">${state.quizScore} out of ${total} correct · ${formatTime(state.quizSeconds)}</div>
            <div class="quiz-nav" style="margin-top:28px;">
                <button class="quiz-nav-btn primary" id="quizRetry">🔄 Retry Quiz</button>
            </div>
        </div>
    `;

    $('#quizRetry').addEventListener('click', renderQuiz);
    
    // Re-bind cursor hover for new button
    $('#quizRetry').addEventListener('mouseenter', () => document.body.classList.add('cursor-hover'));
    $('#quizRetry').addEventListener('mouseleave', () => document.body.classList.remove('cursor-hover'));
}

function fireConfetti() {
    var count = 200;
    var defaults = { origin: { y: 0.7 } };

    function fire(particleRatio, opts) {
        confetti(Object.assign({}, defaults, opts, {
            particleCount: Math.floor(count * particleRatio)
        }));
    }

    fire(0.25, { spread: 26, startVelocity: 55 });
    fire(0.2, { spread: 60 });
    fire(0.35, { spread: 100, decay: 0.91, scalar: 0.8 });
    fire(0.1, { spread: 120, startVelocity: 25, decay: 0.92, scalar: 1.2 });
    fire(0.1, { spread: 120, startVelocity: 45 });
}

// ═══════════════════════════════════════════════
// CHAT
// ═══════════════════════════════════════════════
function renderChatUI() {
    const container = $(`#tab-chat`);
    container.innerHTML = `
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="chat-welcome">
                    <div class="chat-welcome-icon">💬</div>
                    <p>Ask any question about the lecture!</p>
                    <p style="font-size:0.8rem">The AI will answer based on the lecture content.</p>
                </div>
            </div>
            <div class="chat-input-area">
                <input type="text" class="chat-input" id="chatInput" placeholder="Type your question..." autocomplete="off" />
                <button class="chat-send-btn" id="chatSend" title="Send">➤</button>
            </div>
        </div>
    `;

    const input = $('#chatInput');
    const sendBtn = $('#chatSend');

    const send = () => {
        const q = input.value.trim();
        if (!q) return;
        input.value = '';
        sendChatMessage(q);
    };

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') send();
    });
}

async function sendChatMessage(question) {
    const msgs = $('#chatMessages');
    // Remove welcome
    const welcome = msgs.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    // Add user message
    appendChatBubble('user', question);
    state.chatHistory.push({ role: 'user', text: question });

    // Show typing
    const typingEl = document.createElement('div');
    typingEl.className = 'typing-indicator';
    typingEl.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    msgs.appendChild(typingEl);
    msgs.scrollTop = msgs.scrollHeight;

    $('#chatSend').disabled = true;

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: state.currentUrl,
                question: question,
                history: state.chatHistory,
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Chat failed');

        typingEl.remove();
        appendChatBubble('assistant', data.answer);
        state.chatHistory.push({ role: 'assistant', text: data.answer });
    } catch (err) {
        typingEl.remove();
        appendChatBubble('assistant', `⚠️ Error: ${err.message}`);
    }

    $('#chatSend').disabled = false;
    $('#chatInput').focus();
}

function appendChatBubble(role, text) {
    const msgs = $('#chatMessages');
    const bubble = document.createElement('div');
    bubble.className = `chat-message ${role}`;
    bubble.innerHTML = role === 'assistant' ? markdownToHtml(text) : escapeHtml(text);
    msgs.appendChild(bubble);
    msgs.scrollTop = msgs.scrollHeight;
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ═══════════════════════════════════════════════
// ANALYTICS
// ═══════════════════════════════════════════════
function renderAnalytics() {
    const a = state.analysis;
    const container = $(`#tab-analytics`);

    container.innerHTML = `
        <div class="analytics-grid">
            <div class="analytics-card">
                <h3>📊 Topic Difficulty</h3>
                <div class="chart-wrapper"><canvas id="difficultyChart"></canvas></div>
            </div>
            <div class="analytics-card">
                <h3>📈 Lecture Metrics</h3>
                <div class="chart-wrapper"><canvas id="metricsChart"></canvas></div>
            </div>
            <div class="analytics-card full-width">
                <h3>🎯 Overall Analysis</h3>
                <div class="overall-stats">
                    <div class="stat-item">
                        <div class="stat-value">${a.overall.pace}/10</div>
                        <div class="stat-label">Pace</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${a.overall.clarity}/10</div>
                        <div class="stat-label">Clarity</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${a.overall.engagement}/10</div>
                        <div class="stat-label">Engagement</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${a.overall.difficulty}/10</div>
                        <div class="stat-label">Difficulty</div>
                    </div>
                </div>
                <div class="analysis-summary">
                    <strong>Tone:</strong> ${a.overall.tone}<br>
                    <strong>Summary:</strong> ${a.overall.summary}
                </div>
            </div>
        </div>
    `;

    // ─ Topic Difficulty Bar Chart ─
    const topicLabels = a.topics.map(t => t.name);
    const topicDiffs = a.topics.map(t => t.difficulty);

    const ctxBar = document.getElementById('difficultyChart').getContext('2d');
    new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: topicLabels,
            datasets: [{
                label: 'Difficulty',
                data: topicDiffs,
                backgroundColor: topicDiffs.map(d =>
                    d >= 7 ? 'rgba(244,63,94,0.7)' :
                    d >= 4 ? 'rgba(245,158,11,0.7)' :
                    'rgba(16,185,129,0.7)'
                ),
                borderRadius: 6,
                barThickness: 30,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 10, ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted') }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-muted'), maxRotation: 45 }, grid: { display: false } },
            },
            plugins: { legend: { display: false } },
        },
    });

    // ─ Metrics Radar Chart ─
    const ctxRadar = document.getElementById('metricsChart').getContext('2d');
    new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Pace', 'Clarity', 'Engagement', 'Difficulty'],
            datasets: [{
                label: 'Lecture Metrics',
                data: [a.overall.pace, a.overall.clarity, a.overall.engagement, a.overall.difficulty],
                backgroundColor: 'rgba(124,58,237,0.2)',
                borderColor: 'rgba(124,58,237,0.8)',
                pointBackgroundColor: 'rgba(6,182,212,1)',
                pointBorderColor: '#fff',
                pointRadius: 5,
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true, max: 10,
                    ticks: { display: false },
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    pointLabels: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary'), font: { size: 12 } },
                },
            },
            plugins: { legend: { display: false } },
        },
    });
}

// ═══════════════════════════════════════════════
// VIDEO EMBED
// ═══════════════════════════════════════════════
function renderVideoEmbed(videoId) {
    $('#videoEmbed').innerHTML = `
        <iframe src="https://www.youtube.com/embed/${videoId}"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen id="ytPlayer"></iframe>
    `;
}

// ═══════════════════════════════════════════════
// ACTIONS — Copy & PDF Download
// ═══════════════════════════════════════════════
function copySummary() {
    navigator.clipboard.writeText(state.summaryMarkdown).then(() => {
        const btn = $('#copyBtn');
        btn.textContent = '✅ Copied!';
        btn.classList.add('success');
        setTimeout(() => {
            btn.innerHTML = '📋 Copy Summary';
            btn.classList.remove('success');
        }, 2000);
    });
}

function downloadPdf() {
    console.log("Triggering PDF Export...");
    if (window.jspdf && window.jspdf.jsPDF) {
        try {
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            const lines = doc.splitTextToSize(state.summaryMarkdown, 170);
            let y = 20;
            const pageHeight = doc.internal.pageSize.height;

            doc.setFont('helvetica');
            doc.setFontSize(10);

            lines.forEach(line => {
                if (y > pageHeight - 20) {
                    doc.addPage();
                    y = 20;
                }
                doc.text(line, 20, y);
                y += 6;
            });

            doc.save('lecture-summary.pdf');
            return;
        } catch (e) {
            console.error("jsPDF generation failed:", e);
        }
    }
    
    // Fallback native print trick if jsPDF fails or is blocked
    console.log("Using PDF Print Fallback...");
    const printWindow = window.open('', '', 'height=600,width=800');
    printWindow.document.write('<html><head><title>Lecture Summary</title>');
    printWindow.document.write('<style>body{font-family:sans-serif;line-height:1.6;padding:40px;}</style>');
    printWindow.document.write('</head><body>');
    printWindow.document.write($('#summaryContent').innerHTML);
    printWindow.document.write('</body></html>');
    printWindow.document.close();
    printWindow.print();
}

// ═══════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════
function showSkeleton(show) {
    const el = $('#skeletonLoader');
    if (show) {
        el.classList.add('active');
        $('#submitBtn').classList.add('loading');
        $('#submitBtn').disabled = true;
    } else {
        el.classList.remove('active');
        $('#submitBtn').classList.remove('loading');
        $('#submitBtn').disabled = false;
    }
}

function showResults() {
    $('#resultsSection').classList.add('active');
}

function hideResults() {
    $('#resultsSection').classList.remove('active');
}

function showError(msg) {
    const el = $('#errorMsg');
    el.textContent = '⚠️ ' + msg;
    el.style.display = 'block';
}

function hideError() {
    $('#errorMsg').style.display = 'none';
}
