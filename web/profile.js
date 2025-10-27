// --- Backend URL ---
const BACKEND_URL = ''; // Relative path

// --- Default Avatar Path (Served by Frontend/Vercel) ---
const DEFAULT_AVATAR_STATIC_PATH = "/avatars/default_avatar.png";

// --- Elements ---
const avatarPreview = document.getElementById('avatarPreview');
const avatarUpload = document.getElementById('avatarUpload');
const avatarWrapper = document.getElementById('avatarWrapper');
const displayNameInput = document.getElementById('displayNameInput'); // Corrected ID
const statusMessageInput = document.getElementById('statusMessageInput'); // Corrected ID
const saveProfileBtn = document.getElementById('saveProfileBtn'); // Corrected ID
const cancelBtn = document.getElementById('cancelBtn'); // Corrected ID
const saveMessage = document.getElementById('saveMessage'); // Corrected ID

let username = localStorage.getItem('luvisa_user') || null;
let currentAvatarFile = null; // To hold the selected file object for upload

// --- FIX: Matched backend limit of 100KB ---
const MAX_AVATAR_SIZE_KB = 100;

// ---------- Initialization ----------
window.addEventListener('DOMContentLoaded', async () => {
    if (!username) { window.location.href = 'login.html'; return; }
    if (avatarWrapper) avatarWrapper.addEventListener('click', () => avatarUpload.click());
    if (avatarUpload) avatarUpload.addEventListener('change', handleAvatarChange);
    if (saveProfileBtn) saveProfileBtn.addEventListener('click', saveProfileChanges);
    if (cancelBtn) cancelBtn.addEventListener('click', () => window.location.href = 'chat'); // Go to /chat route
    await loadCurrentProfile();
});

// ---------- Load Current Profile ----------
async function loadCurrentProfile() {
    saveMessage.textContent = 'Loading profile...';
    try {
        const response = await fetch(`/api/profile?email=${encodeURIComponent(username)}`); // Relative URL
        const data = await response.json();

        if (response.ok && data.success && data.profile) {
            const profile = data.profile;
            displayNameInput.value = profile.display_name || username.split('@')[0];
            statusMessageInput.value = profile.status || 'Hey there! I’m using Luvisa 💗';

            // --- UPDATED: Handle backend URL or default ---
            if (profile.avatar) {
                // Add a cache-buster query parameter to force browser to reload the image
                avatarPreview.src = `${profile.avatar}?t=${new Date().getTime()}`;
            } else {
                avatarPreview.src = DEFAULT_AVATAR_STATIC_PATH;
            }
            // --- End Update ---

            saveMessage.textContent = '';
        } else {
             console.error('Failed load profile:', data.message);
             saveMessage.textContent = `Error: ${data.message || 'Could not load profile.'}`;
             saveMessage.className = 'save-message error';
             // Set defaults
             displayNameInput.value = username.split('@')[0];
             statusMessageInput.value = 'Hey there! I’m using Luvisa 💗';
             avatarPreview.src = DEFAULT_AVATAR_STATIC_PATH;
        }
    } catch (error) {
        console.error('Load profile network error:', error);
        saveMessage.textContent = 'Network error loading profile.';
        saveMessage.className = 'save-message error';
        // Set defaults
        displayNameInput.value = username.split('@')[0];
        statusMessageInput.value = 'Hey there! I’m using Luvisa 💗';
        avatarPreview.src = DEFAULT_AVATAR_STATIC_PATH;
    }
}

// ---------- Handle Avatar Selection ----------
function handleAvatarChange(event) {
    const file = event.target.files[0];
    saveMessage.textContent = ''; // Clear message
    saveMessage.className = 'save-message';

    if (file && file.type.startsWith("image/")) {
        // Frontend size check
        if (file.size > MAX_AVATAR_SIZE_KB * 1024) {
             saveMessage.textContent = `Image too large! Please choose one under ${MAX_AVATAR_SIZE_KB} KB.`;
             saveMessage.className = 'save-message error';
             avatarUpload.value = ''; // Clear the selection
             currentAvatarFile = null;
             return;
        }

        currentAvatarFile = file; // Store the valid file object
        const reader = new FileReader();
        reader.onload = (e) => {
            avatarPreview.src = e.target.result; // Show preview
        };
        reader.readAsDataURL(file);
    } else {
        currentAvatarFile = null;
        if (file) {
            saveMessage.textContent = 'Please select a valid image file.';
            saveMessage.className = 'save-message error';
        }
    }
}

// ---------- Save Profile Changes ----------
async function saveProfileChanges() {
    const displayName = displayNameInput.value.trim();
    const statusMessage = statusMessageInput.value.trim();

    if (!displayName) { 
        saveMessage.textContent = 'Display name cannot be empty.';
        saveMessage.className = 'save-message error';
        return; 
    }

    saveProfileBtn.classList.add("loading");
    saveProfileBtn.disabled = true;
    saveMessage.textContent = 'Saving...';
    saveMessage.className = 'save-message';

    const formData = new FormData();
    formData.append('email', username);
    formData.append('display_name', displayName);
    formData.append('status_message', statusMessage);
    if (currentAvatarFile) {
        formData.append('avatar_file', currentAvatarFile); // Send the file
    }

    try {
        const response = await fetch(`/api/profile`, { // Relative URL
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (response.ok && data.success) {
            saveMessage.textContent = data.message;
            saveMessage.className = 'save-message success';

            // --- UPDATED: Update preview with URL from response ---
            if (data.profile && data.profile.avatar) {
                 // Add a cache-buster query parameter to force browser to reload the image
                avatarPreview.src = `${data.profile.avatar}?t=${new Date().getTime()}`;
            } else {
                avatarPreview.src = DEFAULT_AVATAR_STATIC_PATH;
            }
            // --- End Update ---

            currentAvatarFile = null; // Reset file state
            avatarUpload.value = ''; // Clear the file input visually
             setTimeout(() => window.location.href = 'chat', 1500); // Go to /chat
        } else {
            if (response.status === 413) {
                 saveMessage.textContent = data.message || `Avatar image is too large (max ${MAX_AVATAR_SIZE_KB} KB).`;
            } else {
                 saveMessage.textContent = `Error: ${data.message || 'Failed to save profile.'}`;
            }
            saveMessage.className = 'save-message error';
        }
    } catch (error) {
        console.error('Save profile network error:', error);
        saveMessage.textContent = 'Network error saving profile.';
        saveMessage.className = 'save-message error';
    } finally {
        saveProfileBtn.classList.remove("loading");
        saveProfileBtn.disabled = false;
    }
}
