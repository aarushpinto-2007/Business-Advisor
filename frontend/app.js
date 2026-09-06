/**
 * app.js — BizSaathi Frontend Application Logic
 * Handles: session management, API calls, message rendering,
 *          profile sidebar updates, Web Speech API, and UI interactions.
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Constants & State
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = window.location.origin; // Same-origin: FastAPI serves both

const state = {
  userId:       null,
  displayName:  null,
  language:     "en",
  isLoading:    false,
  turnCount:    0,
  lastRagHits:  0,
  recognition:  null,   // Web Speech API instance
  isRecording:  false,
};

// ─────────────────────────────────────────────────────────────────────────────
// Session Management
// ─────────────────────────────────────────────────────────────────────────────

function startSession() {
  const userIdInput   = document.getElementById("user-id-input");
  const displayInput  = document.getElementById("user-name-display");

  const rawId = userIdInput.value.trim();
  if (!rawId) {
    userIdInput.style.borderColor = "#FF5050";
    userIdInput.focus();
    setTimeout(() => { userIdInput.style.borderColor = ""; }, 2000);
    return;
  }

  // Sanitise: alphanumeric + underscore + hyphen, max 64 chars
  state.userId      = rawId.replace(/[^a-zA-Z0-9_\-+@ .]/g, "_").slice(0, 64);
  state.displayName = displayInput.value.trim() || state.userId;
  state.language    = document.getElementById("language-select").value;

  // Persist to localStorage for returning users
  localStorage.setItem("bizsaathi_user_id",      state.userId);
  localStorage.setItem("bizsaathi_display_name", state.displayName);
  localStorage.setItem("bizsaathi_language",     state.language);

  document.getElementById("setup-modal").style.display = "none";
  document.getElementById("stat-uid").textContent = state.userId;

  // Update language selector
  document.getElementById("language-select").value = state.language;

  // Load profile + history
  loadProfile();
  loadHistory();

  showToast("✅ Welcome back, " + state.displayName + "!", "success");
}

function checkReturningUser() {
  const savedId   = localStorage.getItem("bizsaathi_user_id");
  const savedName = localStorage.getItem("bizsaathi_display_name");
  const savedLang = localStorage.getItem("bizsaathi_language");

  if (savedId) {
    document.getElementById("user-id-input").value    = savedId;
    document.getElementById("user-name-display").value = savedName || "";
    if (savedLang) document.getElementById("language-select").value = savedLang;
    // Auto-start for returning users
    startSession();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// API Functions
// ─────────────────────────────────────────────────────────────────────────────

async function apiChat(message) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id:  state.userId,
      message:  message,
      language: state.language,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

async function apiGetProfile() {
  const res = await fetch(`${API_BASE}/api/profile/${encodeURIComponent(state.userId)}`);
  if (!res.ok) throw new Error(`Profile fetch failed: HTTP ${res.status}`);
  return res.json();
}

async function apiGetHistory() {
  const res = await fetch(`${API_BASE}/api/history/${encodeURIComponent(state.userId)}`);
  if (!res.ok) throw new Error(`History fetch failed: HTTP ${res.status}`);
  return res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Send Message
// ─────────────────────────────────────────────────────────────────────────────

async function sendMessage() {
  if (state.isLoading || !state.userId) return;

  const input   = document.getElementById("message-input");
  const message = input.value.trim();
  if (!message) return;

  // Clear input + reset height
  input.value = "";
  input.style.height = "";
  document.getElementById("send-btn").disabled = true;

  // Hide welcome screen on first message
  const welcomeScreen = document.getElementById("welcome-screen");
  if (welcomeScreen) welcomeScreen.remove();

  // Append user bubble
  appendMessage("user", message, null, new Date());

  // Show typing indicator
  setTyping(true);
  state.isLoading = true;
  updateHeaderStatus("Typing...");

  try {
    const data = await apiChat(message);

    setTyping(false);
    state.isLoading    = false;
    state.turnCount    = data.turn_count;
    state.lastRagHits  = data.relevant_retrieved_history_count;

    // Update stats
    document.getElementById("stat-turns").textContent = data.turn_count;
    document.getElementById("stat-rag").textContent   = data.relevant_retrieved_history_count;

    // Append bot response
    appendMessage("bot", data.response, data.relevant_retrieved_history_count, new Date());

    // Always refresh the profile sidebar so turn count and any quick facts stay current
    loadProfile();

    // If a background Gemini extraction was triggered, do an extra refresh
    // after a longer delay so the extraction has time to complete and commit
    if (data.profile_updated) {
      setTimeout(() => {
        loadProfile();
        showProfileUpdatedTag();
      }, 4000);
      showToast("🧠 Business profile updated!", "info");
    }

    updateHeaderStatus("Online · Gemini 2.0");

  } catch (err) {
    setTyping(false);
    state.isLoading = false;
    appendMessage("bot", `⚠️ Sorry, I encountered an error: ${err.message}. Please try again.`, 0, new Date());
    updateHeaderStatus("Online · Gemini 2.0");
    showToast("❌ " + err.message, "error");
  }

  scrollToBottom();
}

function sendQuickStart(btn) {
  const text = btn.textContent.trim();
  document.getElementById("message-input").value = text;
  sendMessage();
}

// ─────────────────────────────────────────────────────────────────────────────
// Message Rendering
// ─────────────────────────────────────────────────────────────────────────────

function appendMessage(role, content, ragCount, timestamp) {
  const area = document.getElementById("messages-area");

  // Insert before typing indicator
  const typingEl = document.getElementById("typing-indicator");

  const wrapper = document.createElement("div");
  wrapper.className = `message-wrapper ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "👤" : "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  // RAG tag for bot messages with retrieved context
  if (role === "bot" && ragCount && ragCount > 0) {
    const tag = document.createElement("div");
    tag.className = "rag-tag";
    tag.textContent = `🔍 ${ragCount} memory hit${ragCount > 1 ? "s" : ""} retrieved`;
    bubble.appendChild(tag);
  }

  // Render content with basic markdown-like formatting
  const contentDiv = document.createElement("div");
  contentDiv.innerHTML = formatMessage(content);
  bubble.appendChild(contentDiv);

  // Timestamp
  const timeEl = document.createElement("span");
  timeEl.className = "msg-time";
  timeEl.textContent = formatTime(timestamp);
  bubble.appendChild(timeEl);

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);

  area.insertBefore(wrapper, typingEl);
  scrollToBottom();
}

function formatMessage(text) {
  // Escape HTML first
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic: *text*
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

  // Inline code: `code`
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Numbered lists: lines starting with "1. " — wrap each contiguous block in <ol>
  html = html.replace(/((?:^\d+\. .+$\n?)+)/gm, (block) => {
    const items = block.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");
    return `<ol>${items}</ol>`;
  });

  // Bullet lists: lines starting with "- " or "• " — wrap each contiguous block in <ul>
  html = html.replace(/((?:^[-•] .+$\n?)+)/gm, (block) => {
    const items = block.replace(/^[-•] (.+)$/gm, "<li>$1</li>");
    return `<ul>${items}</ul>`;
  });

  // Line breaks
  html = html.replace(/\n/g, "<br>");

  return html;
}

function formatTime(date) {
  if (!(date instanceof Date)) date = new Date(date);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile Sidebar
// ─────────────────────────────────────────────────────────────────────────────

async function loadProfile() {
  if (!state.userId) return;

  try {
    const data    = await apiGetProfile();
    const profile = data.profile;
    renderProfileSidebar(profile);
  } catch (err) {
    console.warn("Profile load failed:", err);
  }
}

function renderProfileSidebar(profile) {
  const container = document.getElementById("profile-fields");

  const fields = [
    { label: "👤 Name",          value: profile.name },
    { label: "🏪 Business",      value: profile.business_type },
    { label: "📍 Location",      value: profile.location },
    { label: "💰 Turnover",      value: profile.turnover },
    { label: "👥 Employees",     value: profile.employee_count },
    { label: "📋 Goals",         value: profile.goals },
    { label: "⚠️ Challenges",    value: profile.challenges },
  ];

  container.innerHTML = "";

  fields.forEach(({ label, value }) => {
    const div = document.createElement("div");
    div.className = "profile-field";

    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.textContent = label;

    const valueEl = document.createElement("span");
    valueEl.className = value ? "value" : "value empty";
    valueEl.textContent = value || "Not yet known";

    div.appendChild(labelEl);
    div.appendChild(valueEl);
    container.appendChild(div);
  });

  // Schemes
  if (profile.schemes_enrolled) {
    const div = document.createElement("div");
    div.className = "profile-field";

    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.textContent = "🏛️ Gov. Schemes";

    const badgesDiv = document.createElement("div");
    badgesDiv.style.display = "flex";
    badgesDiv.style.flexWrap = "wrap";
    badgesDiv.style.gap = "4px";
    badgesDiv.style.marginTop = "4px";

    profile.schemes_enrolled.split(",").forEach(s => {
      const badge = document.createElement("span");
      badge.className = "profile-badge";
      badge.textContent = s.trim();
      badgesDiv.appendChild(badge);
    });

    div.appendChild(labelEl);
    div.appendChild(badgesDiv);
    container.appendChild(div);
  }

  // Extra facts
  if (profile.extra_facts && Object.keys(profile.extra_facts).length > 0) {
    const div = document.createElement("div");
    div.className = "profile-field";

    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.textContent = "📝 Notes";

    const valueEl = document.createElement("span");
    valueEl.className = "value";
    valueEl.style.fontSize = "12px";
    valueEl.textContent = Object.entries(profile.extra_facts)
      .map(([k, v]) => `${k}: ${v}`)
      .join(" · ");

    div.appendChild(labelEl);
    div.appendChild(valueEl);
    container.appendChild(div);
  }

  // Update turn count stat
  if (profile.turn_count) {
    document.getElementById("stat-turns").textContent = profile.turn_count;
  }
}

function showProfileUpdatedTag() {
  const tag = document.getElementById("profile-updated-tag");
  tag.classList.remove("hidden");
  setTimeout(() => tag.classList.add("hidden"), 4000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load History (on session restore)
// ─────────────────────────────────────────────────────────────────────────────

async function loadHistory() {
  if (!state.userId) return;

  try {
    const data = await apiGetHistory();
    if (data.messages && data.messages.length > 0) {
      // Hide welcome screen
      const welcomeScreen = document.getElementById("welcome-screen");
      if (welcomeScreen) welcomeScreen.remove();

      // Add separator
      const sep = document.createElement("div");
      sep.className = "day-separator";
      const sepSpan = document.createElement("span");
      sepSpan.textContent = "— Previous Conversations —";
      sep.appendChild(sepSpan);
      const typingEl = document.getElementById("typing-indicator");
      document.getElementById("messages-area").insertBefore(sep, typingEl);

      // Render messages (limit to last 30 for performance)
      const msgs = data.messages.slice(-30);
      msgs.forEach(msg => {
        appendMessage(
          msg.role === "user" ? "user" : "bot",
          msg.content,
          0,
          new Date(msg.created_at),
        );
      });

      scrollToBottom();
      showToast(`📜 Loaded ${msgs.length} previous messages`, "info");
    }
  } catch (err) {
    console.warn("History load failed:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// UI Helpers
// ─────────────────────────────────────────────────────────────────────────────

function setTyping(visible) {
  const el = document.getElementById("typing-indicator");
  if (visible) {
    el.classList.add("visible");
  } else {
    el.classList.remove("visible");
  }
}

function scrollToBottom() {
  const area = document.getElementById("messages-area");
  requestAnimationFrame(() => {
    area.scrollTop = area.scrollHeight;
  });
}

function updateHeaderStatus(text) {
  document.getElementById("header-status").textContent = text;
}

function handleInputKeydown(e) {
  const isMobile = window.innerWidth <= 768;
  // Desktop: Enter sends, Shift+Enter = newline
  // Mobile: Enter always = newline (soft keyboard)
  if (e.key === "Enter" && !e.shiftKey && !isMobile) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";

  // Enable send button when there's content
  const sendBtn = document.getElementById("send-btn");
  sendBtn.disabled = textarea.value.trim().length === 0 || state.isLoading;
}

function clearChat() {
  if (!confirm("Start a new conversation? Chat history on this device will be cleared.")) return;

  // Remove all messages except typing indicator and welcome screen
  const area = document.getElementById("messages-area");
  const typingEl = document.getElementById("typing-indicator");
  const children = [...area.children];
  children.forEach(child => {
    if (child.id !== "typing-indicator" && child.id !== "welcome-screen") {
      child.remove();
    }
  });

  // Recreate welcome screen
  if (!document.getElementById("welcome-screen")) {
    const ws = createWelcomeScreen();
    area.insertBefore(ws, typingEl);
  }

  showToast("🗑️ Chat cleared. Start fresh!", "info");
}

function createWelcomeScreen() {
  const div = document.createElement("div");
  div.id = "welcome-screen";
  div.innerHTML = `
    <div class="welcome-icon">🌱</div>
    <h2 class="welcome-title">Namaste! I'm BizSaathi</h2>
    <p class="welcome-subtitle">Ask me anything about your business, government schemes, loans, GST, or marketing.</p>
    <div class="quick-starts">
      <button class="quick-start-btn" onclick="sendQuickStart(this)">💰 Mudra Loan kaise milega?</button>
      <button class="quick-start-btn" onclick="sendQuickStart(this)">📊 Help me with GST registration</button>
      <button class="quick-start-btn" onclick="sendQuickStart(this)">🏪 How to grow my small shop?</button>
      <button class="quick-start-btn" onclick="sendQuickStart(this)">📱 UPI payment setup karo</button>
    </div>
  `;
  return div;
}

// ─────────────────────────────────────────────────────────────────────────────
// Language Selector
// ─────────────────────────────────────────────────────────────────────────────

function onLanguageChange() {
  const newLang = document.getElementById("language-select").value;
  state.language = newLang;
  localStorage.setItem("bizsaathi_language", newLang);

  const langNames = {
    en: "English", hi: "Hindi", mr: "Marathi",
    ta: "Tamil",   te: "Telugu", kn: "Kannada",
    bn: "Bengali", gu: "Gujarati",
  };

  showToast(`🌐 Language switched to ${langNames[newLang] || newLang}`, "info");
}

// ─────────────────────────────────────────────────────────────────────────────
// Web Speech API (Voice Input)
// ─────────────────────────────────────────────────────────────────────────────

const SPEECH_LANG_MAP = {
  en: "en-IN", hi: "hi-IN", mr: "mr-IN",
  ta: "ta-IN", te: "te-IN", kn: "kn-IN",
  bn: "bn-IN", gu: "gu-IN",
};

function toggleMic() {
  if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
    showToast("🎙️ Voice input not supported in this browser. Try Chrome.", "error");
    return;
  }

  if (state.isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

function startRecording() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  state.recognition = new SpeechRecognition();

  state.recognition.lang        = SPEECH_LANG_MAP[state.language] || "en-IN";
  state.recognition.interimResults = true;
  state.recognition.maxAlternatives = 1;

  state.recognition.onstart = () => {
    state.isRecording = true;
    document.getElementById("mic-btn").classList.add("recording");
    document.getElementById("mic-btn").textContent = "🔴";
    document.getElementById("message-input").placeholder = "Listening...";
    showToast("🎙️ Listening... Speak now", "info");
  };

  state.recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map(r => r[0].transcript)
      .join("");
    const input = document.getElementById("message-input");
    input.value = transcript;
    autoResize(input);
  };

  state.recognition.onerror = (event) => {
    console.warn("Speech error:", event.error);
    stopRecording();
    if (event.error !== "aborted") {
      showToast("🎙️ Voice input error: " + event.error, "error");
    }
  };

  state.recognition.onend = () => {
    stopRecording();
    // Auto-send if we got a transcript
    const val = document.getElementById("message-input").value.trim();
    if (val) sendMessage();
  };

  state.recognition.start();
}

function stopRecording() {
  state.isRecording = false;
  document.getElementById("mic-btn").classList.remove("recording");
  document.getElementById("mic-btn").textContent = "🎙️";
  document.getElementById("message-input").placeholder = "Type your business question...";
  if (state.recognition) {
    try { state.recognition.stop(); } catch (_) {}
    state.recognition = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar Toggle (Mobile)
// ─────────────────────────────────────────────────────────────────────────────

function toggleSidebar() {
  const sidebar  = document.getElementById("sidebar");
  const overlay  = document.getElementById("sidebar-overlay");
  const isOpen   = sidebar.classList.contains("open");

  if (isOpen) {
    closeSidebar();
  } else {
    sidebar.classList.add("open");
    overlay.classList.add("visible");
    document.body.style.overflow = "hidden";
  }
}

function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("sidebar-overlay").classList.remove("visible");
  document.body.style.overflow = "";
}

// ─────────────────────────────────────────────────────────────────────────────
// Toast Notifications
// ─────────────────────────────────────────────────────────────────────────────

function showToast(message, type = "info", duration = 3000) {
  const container = document.getElementById("toast-container");

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;

  container.appendChild(toast);

  // Auto-remove
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px) scale(0.95)";
    toast.style.transition = "all 0.25s ease";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  // Keyboard shortcut: Enter on modal inputs
  document.getElementById("user-id-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") startSession();
  });
  document.getElementById("user-name-display").addEventListener("keydown", (e) => {
    if (e.key === "Enter") startSession();
  });

  // Message input: enable/disable send button
  document.getElementById("message-input").addEventListener("input", (e) => {
    autoResize(e.target);
  });

  // Check for returning user
  checkReturningUser();
});
