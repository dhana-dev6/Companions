// --- NEW: Define Backend URL ---
const BACKEND_URL = ''; // Use relative path

document.addEventListener("DOMContentLoaded", () => {
    handleAutoLogin(); // Runs automatically

    // Attach events for manual login/signup
    const loginBtn = document.getElementById("loginBtn");
    const signupBtn = document.getElementById("signupBtn");

    if (loginBtn) {
        loginBtn.addEventListener("click", loginUser);
        document.getElementById("loginPass").addEventListener("keydown", (e) => {
            if (e.key === 'Enter') loginUser();
        });
    }

    if (signupBtn) {
        signupBtn.addEventListener("click", signupUser);
        document.getElementById("signupPass").addEventListener("keydown", (e) => {
            if (e.key === 'Enter') signupUser();
        });
    }
});

async function handleAutoLogin() {
    const savedUser = localStorage.getItem("luvisa_user");
    if (!savedUser) {
        setRandomBackground(); // Set background only if not auto-logging in
        return;
    }

    // Show "Logging in" message
    const authContainer = document.querySelector(".auth-container");
    if (authContainer) {
        authContainer.innerHTML = `
            <div class="auto-login-view">
                <img src="luvisa.png" alt="Luvisa" class="luvisa-logo">
                <h2>Welcome Back!</h2>
                <p>Logging you in securely...</p>
                <div class="spinner"></div>
            </div>
        `;
    }

    try {
        // --- UPDATED: Use fetch with relative URL ---
        const response = await fetch(`${BACKEND_URL}/api/auto_login_check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: savedUser })
        });
        const data = await response.json();

        if (response.ok && data.isValid) {
            window.location.href = "/chat"; // Go to /chat route
        } else {
            console.warn("Auto-login failed:", data.message || 'User not valid');
            localStorage.removeItem("luvisa_user");
            window.location.reload(); // Show login form
        }
    } catch (error) {
        console.error("Auto-login check error:", error);
        localStorage.removeItem("luvisa_user");
        alert("Could not verify login. Please log in manually.");
        window.location.reload(); // Show login form
    }
}

function setRandomBackground() {
    const backgrounds = [
        "backgrounds/bg1.jpg", "backgrounds/bg6.jpg", "backgrounds/bg3.jpg",
        "backgrounds/bg4.jpg", "backgrounds/bg5.jpg"
    ];
    const randomBg = backgrounds[Math.floor(Math.random() * backgrounds.length)];
    document.body.style.backgroundImage = `url('${randomBg}')`;
}

async function loginUser() {
    const email = document.getElementById("loginEmail").value.trim();
    const password = document.getElementById("loginPass").value.trim();
    const msg = document.getElementById("loginMsg");
    const btn = document.getElementById("loginBtn");

    if (!email || !password) {
        msg.textContent = "Please fill in all fields.";
        return;
    }

    btn.classList.add("loading");
    btn.disabled = true;
    msg.textContent = ""; 

    try {
        // --- UPDATED: Use fetch with relative URL ---
        const response = await fetch(`${BACKEND_URL}/api/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email, password: password })
        });
        const data = await response.json();

        msg.textContent = data.message; 

        if (response.ok && data.success) {
            localStorage.setItem("luvisa_user", data.email); 
            window.location.href = "/chat"; // Go to /chat route
        } else {
             msg.style.color = "#ffc2d1"; 
        }

    } catch (error) {
        console.error("Login error:", error);
        msg.textContent = "Login failed. Could not connect to server.";
        msg.style.color = "#ffc2d1";
    } finally {
        btn.classList.remove("loading");
        btn.disabled = false;
    }
}

async function signupUser() {
    const email = document.getElementById("signupEmail").value.trim();
    const password = document.getElementById("signupPass").value.trim();
    const msg = document.getElementById("signupMsg");
    const btn = document.getElementById("signupBtn");

    if (email.length < 5 || password.length < 4) {
        msg.textContent = "Please enter a valid email and a password of at least 4 characters.";
        return;
    }

    btn.classList.add("loading");
    btn.disabled = true;
    msg.textContent = ""; 

    try {
         // --- UPDATED: Use fetch with relative URL ---
        const response = await fetch(`${BACKEND_URL}/api/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: email, password: password })
        });
        const data = await response.json();

        msg.textContent = data.message;

        if (response.ok && data.success) {
            msg.style.color = "#a7c957"; 
            setTimeout(() => {
                window.location.href = "/login"; // Go to /login route
            }, 1500);
        } else {
            msg.style.color = "#ffc2d1";
        }

    } catch (error) {
        console.error("Signup error:", error);
        msg.textContent = "Signup failed. Could not connect to server.";
        msg.style.color = "#ffc2d1";
    } finally {
        btn.classList.remove("loading");
        btn.disabled = false;
    }
}