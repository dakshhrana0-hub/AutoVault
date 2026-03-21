document.addEventListener("DOMContentLoaded", function() {
    // 1. Inject the inner HTML for #navAuth if it's empty
    const navAuth = document.getElementById('navAuth');
    if (navAuth && navAuth.innerHTML.trim() === '') {
        navAuth.innerHTML = `
          <a href="login.html" class="nav-profile-btn" id="navProfileBtn">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          </a>
          <div class="nav-avatar" id="navAvatar">
            <div class="nav-avatar-circle" id="navInitial">?</div>
            <span class="nav-avatar-name" id="navName">GUEST</span>
            <div class="nav-avatar-dropdown">
              <a href="profile.html">My Profile</a>
              <a href="compare.html">Compare Cars</a>
              <a href="#" class="logout" id="logoutBtn">Sign Out</a>
            </div>
          </div>
        `;
    }

    // 2. Initialize Supabase if not already
    var sb = window.supabaseClient || null; // Some pages use window.supabaseClient
    if (!sb && window.supabase && typeof SUPABASE_URL !== 'undefined') {
        try {
            sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON, {
              auth: { autoRefreshToken: true, persistSession: true, detectSessionInUrl: false }
            });
        } catch(e) {}
    }

    if (!sb) return;

    // 3. Auth Logic
    var navAvatar    = document.getElementById('navAvatar');
    var navInitial   = document.getElementById('navInitial');
    var navName      = document.getElementById('navName');
    var navProfileBtn= document.getElementById('navProfileBtn');
    var logoutBtn    = document.getElementById('logoutBtn');

    async function fetchProfileName(userId) {
      try {
        var r = await sb.from('profiles').select('first_name, last_name').eq('id', userId).single();
        return r.data || null;
      } catch(e) { return null; }
    }

    async function updateNavUser(user) {
      if (!navInitial) return;
      var p = await fetchProfileName(user.id);
      var meta = user.user_metadata || {};
      var first = (p && p.first_name) ? p.first_name : (meta.first_name || user.email.split('@')[0]);
      var last  = (p && p.last_name)  ? p.last_name  : (meta.last_name  || '');
      var initials = ((first[0]||'') + (last[0]||'')).toUpperCase() || first[0].toUpperCase();

      navInitial.textContent = initials;
      navName.textContent    = first;
      navAvatar.classList.add('show');
      if (navProfileBtn) navProfileBtn.style.display = 'none';
    }

    function clearNavUser() {
      if (navAvatar) navAvatar.classList.remove('show');
      if (navProfileBtn) navProfileBtn.style.display = '';
    }

    sb.auth.getSession().then(function(r) {
      if (r.data && r.data.session && r.data.session.user) updateNavUser(r.data.session.user);
    }).catch(function(){});

    sb.auth.onAuthStateChange(function(evt, session) {
      if (session && session.user) updateNavUser(session.user);
      else clearNavUser();
    });

    if (logoutBtn) {
      logoutBtn.addEventListener('click', async function(e) {
        e.preventDefault();
        await sb.auth.signOut().catch(function(){});
        clearNavUser();
        if(window.location.pathname.includes('profile')) {
            window.location.href = 'login.html';
        } else {
            window.location.reload();
        }
      });
    }
});
