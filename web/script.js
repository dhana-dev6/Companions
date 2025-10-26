// --- Backend URL ---
const BACKEND_URL = ''; // Relative path

// --- Default Avatar Path ---
const DEFAULT_AVATAR_STATIC_PATH = "/avatars/default_avatar.png";

// --- Elements ---
const luvisaProfilePic = document.getElementById('luvisaProfilePic');
const closeLuvisaProfileBtn = document.getElementById('closeLuvisaProfileBtn');
const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const userAvatarHeader = document.getElementById('userAvatarHeader');
const userAvatarWrapper = document.getElementById('userAvatarWrapper');
const dropdown = document.getElementById('profileDropdown');
const dropdownAvatar = document.getElementById('dropdownAvatar');
const dropdownName = document.getElementById('dropdownName');
const dropdownStatus = document.getElementById('dropdownStatus');
const headerUserName = document.getElementById('headerUserName');
const notifySound = document.getElementById('notifySound');
const menuIcon = document.getElementById('menuIcon');
const sidebarMenu = document.getElementById('sidebarMenu');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const closeSidebarIcon = document.getElementById('closeSidebarIcon');
const luvisaProfilePanel = document.getElementById('luvisaProfilePanel');

// --- NEW SETTINGS ELEMENTS ---
const settingsToggle = document.getElementById('settingsToggle');
const settingsSubmenu = document.getElementById('settingsSubmenu');
const editProfileBtn = document.getElementById('editProfileBtn'); // Now in submenu
const forgetBtn = document.getElementById('forgetBtn'); // Now in submenu
const logoutBtn = document.getElementById('logoutBtn'); // Still in dropdown
// --- END ---


let username = localStorage.getItem('luvisa_user') || null;

// --- Background Setting ---
document.addEventListener("DOMContentLoaded", () => { setRandomBackground(); });
function setRandomBackground() {
    const backgrounds = [ "backgrounds/bg1.jpg", "backgrounds/bg2.jpg", "backgrounds/bg3.jpg", "backgrounds/bg4.jpg", "backgrounds/bg5.jpg","backgrounds/bg6.jpg" ];
    const randomBg = backgrounds[Math.floor(Math.random() * backgrounds.length)];
    if (chatbox) chatbox.style.backgroundImage = `url('${randomBg}')`;
}

// ---------- Initialization ----------
window.addEventListener('DOMContentLoaded', async () => {
    if (!username) { window.location.href = 'login.html'; return; }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });

    // User Dropdown Logic
    if (userAvatarWrapper) {
        userAvatarWrapper.addEventListener('click', (ev) => { ev.stopPropagation(); toggleDropdown(true); });
    }
    document.addEventListener('click', (ev) => {
        if (dropdown && !dropdown.contains(ev.target) && userAvatarWrapper && !userAvatarWrapper.contains(ev.target)) {
            toggleDropdown(false);
        }
    });

    // --- UPDATED: Button listeners target new locations ---
    if(editProfileBtn) {
        editProfileBtn.addEventListener('click', () => {
            toggleSidebar(false); // Close sidebar first
            window.location.href = 'profile.html';
        });
    }
    if(forgetBtn) {
        forgetBtn.addEventListener('click', async () => {
            toggleSidebar(false); // Close sidebar first
            if (!confirm('Do you want to really forget the conversation😥?')) return;
            try {
                const response = await fetch(`/api/forget_memory`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email: username }) });
                const data = await response.json();
                alert(data.message); if (response.ok && data.success) chatbox.innerHTML = '';
            } catch (e) { console.error('Forget err:', e); alert('Could not forget.'); }
        });
    }
    if(logoutBtn) { // Listener for logout in dropdown
        logoutBtn.addEventListener('click', () => {
            toggleDropdown(false); // Close dropdown
            if (!confirm('Logout?')) return;
            localStorage.removeItem('luvisa_user'); window.location.href = 'login.html';
        });
    }
    // --- END UPDATES ---

    // Luvisa Profile Panel Logic
    if (luvisaProfilePic) {
        luvisaProfilePic.addEventListener('click', (e) => { e.stopPropagation(); toggleLuvisaProfile(true); });
    }
    if (closeLuvisaProfileBtn) {
        closeLuvisaProfileBtn.addEventListener('click', () => toggleLuvisaProfile(false));
    }

    // Slide-out Sidebar Logic
    if (menuIcon) {
        menuIcon.addEventListener('click', (e) => {
            e.stopPropagation();
            const shouldShow = sidebarMenu ? !sidebarMenu.classList.contains('show') : false;
            toggleSidebar(shouldShow);
        });
    }
    if (closeSidebarIcon) {
        closeSidebarIcon.addEventListener('click', (e) => { e.stopPropagation(); toggleSidebar(false); });
    }

    // Overlay listener - does nothing due to CSS pointer-events: none
    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', () => {
             // Click passes through
        });
    }

    // Escape key closes BOTH panels and Dropdown
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
             toggleSidebar(false);
             toggleLuvisaProfile(false);
             toggleDropdown(false);
        }
    });

    // --- NEW: Settings Submenu Toggle ---
    if (settingsToggle && settingsSubmenu) {
        settingsToggle.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent sidebar closing if overlay becomes clickable
            settingsToggle.classList.toggle('active');
            settingsSubmenu.classList.toggle('show');
        });
    }
    // --- END ---

    // Initial Load
    await loadAndApplyProfile(username);
    await loadChatHistory(username);
    setTimeout(() => userInput.focus(), 200);
});

function toggleDropdown(show) {
    if (dropdown) {
        dropdown.classList.toggle('show', show);
        dropdown.setAttribute('aria-hidden', !show);
    }
}

// --- UPDATED: Toggle Function for Profile Panel (Push Effect) ---
function toggleLuvisaProfile(show) {
    const body = document.body;
    if (luvisaProfilePanel) {
        luvisaProfilePanel.classList.toggle('show', show); // Toggles slide animation (transform)
        body.classList.toggle('profile-panel-open', show); // Toggles push class on body
    }
     // Optional: Close the main sidebar if opening the profile panel
     if (show && sidebarMenu && sidebarMenu.classList.contains('show')) {
         toggleSidebar(false);
     }
     // Update overlay based on combined state
     updateOverlayVisibility();
}
// --- END UPDATE ---

// --- UPDATED: Slide-out Sidebar Toggle Function (Push Effect) ---
function toggleSidebar(show) {
    const body = document.body; // Get the body element
    if (sidebarMenu) {
        sidebarMenu.classList.toggle('show', show); // Toggles slide animation (transform)
        body.classList.toggle('sidebar-open', show); // Toggles push class on body
    }
    // Update overlay based on combined state
    updateOverlayVisibility();

     // Optional: Close the profile panel if opening the main sidebar
     if (show && luvisaProfilePanel && luvisaProfilePanel.classList.contains('show')) {
         toggleLuvisaProfile(false);
     }
}
// --- END UPDATE ---

// --- Controls Overlay Visibility (Visual Only) ---
function updateOverlayVisibility() {
    const sidebarIsOpen = sidebarMenu && sidebarMenu.classList.contains('show');
    const profilePanelIsOpen = luvisaProfilePanel && luvisaProfilePanel.classList.contains('show');
    // Show overlay if EITHER panel is open
    const shouldShowOverlay = sidebarIsOpen || profilePanelIsOpen;

    if (sidebarOverlay) {
        sidebarOverlay.classList.toggle('show', shouldShowOverlay);
    }
}
// ---------- Profile / header ----------
async function loadAndApplyProfile(user) {
    try {
        const response = await fetch(`/api/profile?email=${encodeURIComponent(user)}`);
        const data = await response.json();

        if (response.ok && data.success && data.profile) {
            const profile = data.profile;
            const displayName = profile.display_name || user.split('@')[0];

            if (profile.avatar) {
                const avatarUrl = profile.avatar + '?t=' + new Date().getTime();
                if (dropdownAvatar) dropdownAvatar.src = avatarUrl;
                if (userAvatarHeader) userAvatarHeader.src = avatarUrl;
            } else {
                if (dropdownAvatar) dropdownAvatar.src = DEFAULT_AVATAR_STATIC_PATH;
                if (userAvatarHeader) userAvatarHeader.src = DEFAULT_AVATAR_STATIC_PATH;
            }

            if (headerUserName) headerUserName.textContent = displayName;
            if (dropdownName) dropdownName.textContent = displayName;
            if (dropdownStatus) dropdownStatus.textContent = profile.status || 'Online';
        } else {
             console.error('Failed load profile:', data.message);
             const defaultName = user.split('@')[0];
             if (headerUserName) headerUserName.textContent = defaultName;
             if (dropdownName) dropdownName.textContent = defaultName;
             if (dropdownStatus) dropdownStatus.textContent = 'Online';
             if (dropdownAvatar) dropdownAvatar.src = DEFAULT_AVATAR_STATIC_PATH;
             if (userAvatarHeader) userAvatarHeader.src = DEFAULT_AVATAR_STATIC_PATH;
        }
    } catch (err) {
        console.error('Load profile network error:', err);
         const defaultName = user.split('@')[0];
         if (headerUserName) headerUserName.textContent = defaultName;
         if (dropdownName) dropdownName.textContent = defaultName;
         if (dropdownStatus) dropdownStatus.textContent = 'Offline?';
         if (dropdownAvatar) dropdownAvatar.src = DEFAULT_AVATAR_STATIC_PATH;
         if (userAvatarHeader) userAvatarHeader.src = DEFAULT_AVATAR_STATIC_PATH;
    }
}

// ---------- Chat history ----------
async function loadChatHistory(user) {
    try {
        const response = await fetch(`/api/chat_history?email=${encodeURIComponent(user)}`);
        const data = await response.json();

        if (response.ok && data.success) {
            chatbox.innerHTML = '';
            data.history.forEach(m => appendMessage(m.sender === 'user' ? 'user' : 'luvisa', m.message, m.time));
            chatbox.scrollTop = chatbox.scrollHeight;
        } else {
            console.error('Failed load history:', data.message); appendMessage('luvisa', "Couldn't load messages 😥");
        }
    } catch (err) {
        console.error('Load history network error:', err); appendMessage('luvisa', "Error loading history 💔");
    }
}

// ---------- Append message helper ----------
function appendMessage(type, text, atTime = null) {
    const wrapper = document.createElement('div'); wrapper.className = `message ${type}-message`;
    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    const msg = document.createElement('div'); msg.className = 'message-text'; msg.textContent = text;
    const timeDiv = document.createElement('div'); timeDiv.className = 'message-time'; timeDiv.textContent = formatTime(atTime);
    bubble.appendChild(msg);
    bubble.appendChild(timeDiv);
    wrapper.appendChild(bubble);
    chatbox.appendChild(wrapper);
    chatbox.scrollTop = chatbox.scrollHeight; return wrapper;
}

// ---------- Format Time helper ----------
function formatTime(atTime) {
    if (!atTime) return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    try {
        const dateStr = atTime.includes('T') ? atTime : atTime.replace(' ', 'T') + 'Z';
        const dt = new Date(dateStr);
        if (isNaN(dt.getTime())) return atTime;
        return dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
        console.warn("Error parsing timestamp:", atTime, e);
        return atTime;
    }
}


// ---------- Typing indicator ----------
function showTypingBubble() {
    const wrap = document.createElement('div'); wrap.className = 'message luvisa-message typing-message';
    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    bubble.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
    wrap.appendChild(bubble); chatbox.appendChild(wrap); chatbox.scrollTop = chatbox.scrollHeight; return wrap;
}

// ---------- Send flow ----------
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || !username) return;

    userInput.value = '';
    appendMessage('user', text);
    const typing = showTypingBubble();

    try {
        const response = await fetch(`/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: username, text: text })
        });
        const data = await response.json();

        if (typing?.parentNode) typing.parentNode.removeChild(typing);

        if (response.ok && data.success) {
            appendMessage('luvisa', data.reply);
            if (notifySound) notifySound.play().catch(e => console.warn("Audio err:", e));
        } else {
            console.error('Reply err:', data.message); appendMessage('luvisa', data.message || "Sorry... 💔");
        }
    } catch (err) {
        if (typing?.parentNode) typing.parentNode.removeChild(typing);
        appendMessage('luvisa', "Sorry, connection trouble 😥");
        console.error('Send network error:', err);
    }
}