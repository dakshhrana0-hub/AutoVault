import os
import re

nav_html = """  <nav class="aura-nav">
    <a href="index.html" class="logo hover-target">AutoVault</a>
    <ul class="nav-links">
      <li><a href="all_listing.html" class="hover-target">Models</a></li>
      <li><a href="compare.html" class="hover-target">Compare</a></li>
      <li><a href="prestige_collection.html" class="hover-target">Prestige</a></li>
      
      <!-- Authentic Supabase Login Icon -->
      <li>
        <a class="nav-profile-btn" href="login.html" id="navProfileBtn" title="Sign In">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke-width="1.5">
            <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
          </svg>
        </a>
      </li>

      <!-- Supabase Auth Avatar Dropdown -->
      <li>
        <div class="nav-avatar" id="navAvatar">
          <div class="nav-avatar-circle" id="navInitial">?</div>
          <span class="nav-avatar-name" id="navName" style="display:none;">Driver</span>
          <div class="nav-avatar-dropdown">
            <a href="profile.html">My Profile</a>
            <a href="profile.html#saved">Saved Cars</a>
            <a href="#" class="logout" id="logoutBtn">Sign Out</a>
          </div>
        </div>
      </li>
    </ul>
  </nav>"""

pattern = re.compile(r'<nav\s+class="aura-nav".*?</nav>', re.DOTALL)

for file in os.listdir('.'):
    if file.endswith('.html'):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the first occurrence
        new_content = pattern.sub(nav_html, content, count=1)
        
        if new_content != content:
            with open(file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated nav in {file}")

print("Done")
