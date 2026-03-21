import re
import sys

file_path = "c:/Users/hanni/Desktop/Documents/GitHub/AutoVault/frontend/all_listing.html"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Revert CSS to CSS Grid
    content = re.sub(
        r'\.inventory-wrapper.*?\.showroom-grid\s*{\s*display:\s*flex;.*?}',
        r'.inventory-wrapper { width: 100%; position: relative; }\n        .showroom-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 35px; max-width: 1400px; margin: 60px auto; padding: 0 40px; min-height: 50vh; }',
        content,
        flags=re.DOTALL
    )
    
    # 2. Fix card CSS width
    content = re.sub(
        r'\.car-card\s*{\s*flex:\s*0 0 350px;\s*width:\s*350px;\s*background:',
        r'.car-card {\n            background:',
        content
    )
    # Also remove any transform or scale hover from CSS since JS will handle it smoothly
    content = re.sub(
        r'\.car-card:hover\s*{\s*transform:\s*translateY\(-12px\).*?}',
        r'.car-card:hover { z-index: 10; cursor: none; }',
        content
    )

    # 3. Replace JS logic for scroll and animations
    gsap_replace = """
                bindHoverTargets();

                ScrollTrigger.refresh();
                
                // Clear any previous horizontal scrollTrigger
                if (window.horScroll) {
                    window.horScroll.kill();
                }

                if (this.filteredCars.length > 0) {
                    // Cinematic 3D Entrance Animation
                    gsap.fromTo(this.grid.querySelectorAll('.reveal-chunk'), 
                        { y: 150, opacity: 0, scale: 0.85, rotationX: 25, filter: "blur(12px)" },
                        { 
                            y: 0, opacity: 1, scale: 1, rotationX: 0, filter: "blur(0px)",
                            duration: 1.2, 
                            stagger: 0.08, 
                            ease: "power3.out", 
                            scrollTrigger: { 
                                trigger: this.grid, 
                                start: "top bottom-=50" 
                            },
                            onComplete: () => {
                                this.grid.querySelectorAll('.reveal-chunk').forEach(el => el.classList.remove('reveal-chunk'));
                            }
                        }
                    );
                    
                    // Immersive 3D Tilt Hover
                    this.grid.querySelectorAll('.car-card:not(.tilt-bound)').forEach(card => {
                        card.classList.add('tilt-bound');
                        card.addEventListener('mousemove', e => {
                            let rect = card.getBoundingClientRect();
                            let x = e.clientX - rect.left;
                            let y = e.clientY - rect.top;
                            let deltaX = (x - rect.width / 2) / (rect.width / 2);
                            let deltaY = (y - rect.height / 2) / (rect.height / 2);
                            
                            gsap.to(card, {
                                rotationY: deltaX * 12,
                                rotationX: deltaY * -12,
                                scale: 1.05,
                                duration: 0.4,
                                ease: "power2.out",
                                transformPerspective: 1200,
                                boxShadow: `${-deltaX * 30}px ${deltaY * 30 + 30}px 60px rgba(0,0,0,0.8), 0 0 20px rgba(255,51,0,0.4)`
                            });
                        });
                        card.addEventListener('mouseleave', () => {
                            gsap.to(card, {
                                rotationY: 0,
                                rotationX: 0,
                                scale: 1,
                                duration: 0.7,
                                ease: "power3.out",
                                boxShadow: `0 10px 40px rgba(0,0,0,0.8)`
                            });
                        });
                    });
                }"""
                
    # Use re.sub to match the old JS block
    content = re.sub(
        r'bindHoverTargets\(\);\s*ScrollTrigger\.refresh\(\);\s*// Clear any previous horizontal scrollTrigger.*?}\s*\);\s*}\s*',
        gsap_replace,
        content,
        flags=re.DOTALL
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Success")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
