# homescreen.py

import os
import logging
import zipfile
import plistlib
import datetime
import pygame


class IOSHomeScreen:

    def __init__(self, width=320, height=480):
        self.width = width
        self.height = height
        self.apps_dir = os.path.join(os.getcwd(), "apps")
        os.makedirs(self.apps_dir, exist_ok=True)
        self.apps_metadata = {}
        
        # Pagination & Swipe State
        self.current_page = 0
        self.apps_per_page = 16
        self.drag_start_x = None

        self.scan_apps_folder()

    def scan_apps_folder(self):
        """Scans the 'apps' folder for .ipa files, extracts metadata and the real iTunesArtwork icon."""
        self.apps_metadata.clear()
        if not os.path.exists(self.apps_dir):
            return

        for item in os.listdir(self.apps_dir):
            if item.startswith('.'):
                continue
            item_path = os.path.join(self.apps_dir, item)
            
            if item.endswith('.ipa') and os.path.isfile(item_path):
                app_info = self._parse_ipa(item_path)
                if app_info:
                    self.apps_metadata[app_info['name']] = app_info

    def _parse_ipa(self, ipa_path):
        try:
            with zipfile.ZipFile(ipa_path, 'r') as z:
                plist_name = None
                app_folder = None
                artwork_filename = None
                
                for filename in z.namelist():
                    if 'Payload/' in filename and filename.endswith('.app/Info.plist'):
                        plist_name = filename
                        app_folder = filename.split('Info.plist')[0]
                    if 'itunesartwork' in filename.lower():
                        artwork_filename = filename
                
                raw_file_base = os.path.splitext(os.path.basename(ipa_path))[0]
                # Windows can't create/reach a directory whose name ends in a space or dot
                # ("...iPhone OS 2.2.1 " -> WinError 3 on <dir>\\Payload).
                safe_base = raw_file_base.strip().rstrip('. ') or 'app'
                app_name = raw_file_base
                executable_name = raw_file_base
                
                if plist_name:
                    with z.open(plist_name) as pf:
                        plist_data = plistlib.load(pf)
                        app_name = plist_data.get('CFBundleDisplayName', plist_data.get('CFBundleName', raw_file_base))
                        executable_name = plist_data.get('CFBundleExecutable', raw_file_base)

                binary_path = None
                if app_folder:
                    bin_internal_path = app_folder + executable_name
                    if bin_internal_path not in z.namelist():
                        for file in z.namelist():
                            if file.startswith(app_folder) and not file.endswith('/') and '.' not in os.path.basename(file):
                                bin_internal_path = file
                                break

                    if bin_internal_path in z.namelist():
                        extracted_dir = os.path.join(self.apps_dir, "extracted", safe_base)
                        os.makedirs(extracted_dir, exist_ok=True)
                        binary_path = z.extract(bin_internal_path, extracted_dir)

                icon_surface = None
                if artwork_filename:
                    icon_extract_dir = os.path.join(self.apps_dir, "extracted_icons")
                    os.makedirs(icon_extract_dir, exist_ok=True)
                    extracted_icon_path = z.extract(artwork_filename, icon_extract_dir)
                    try:
                        raw_icon = pygame.image.load(extracted_icon_path).convert_alpha()
                        scaled_icon = pygame.transform.smoothscale(raw_icon, (56, 56))
                        
                        icon_surface = pygame.Surface((56, 56), pygame.SRCALPHA)
                        mask_rect = pygame.Rect(0, 0, 56, 56)
                        
                        pygame.draw.rect(icon_surface, (255, 255, 255), mask_rect, border_radius=12)
                        icon_surface.blit(scaled_icon, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                        
                        gloss_surf = pygame.Surface((56, 28), pygame.SRCALPHA)
                        pygame.draw.rect(gloss_surf, (255, 255, 255, 75), (0, 0, 56, 28), border_top_left_radius=12, border_top_right_radius=12)
                        icon_surface.blit(gloss_surf, (0, 0))

                    except Exception as ie:
                        logging.warning(f"[Icon Loader Warning] Could not mask artwork: {ie}")

                return {
                    'name': app_name,
                    'type': 'ipa',
                    'path': ipa_path,
                    'binary_path': binary_path,
                    'icon': icon_surface
                }
        except Exception as e:
            logging.error(f"[Loader Error] {e}")
            return None

    def total_pages(self):
        total_apps = len(self.apps_metadata)
        if total_apps == 0:
            return 1
        return (total_apps + self.apps_per_page - 1) // self.apps_per_page

    def handle_event(self, event):
        """Handles swipe gestures, arrow keys, and click navigation."""
        max_pages = self.total_pages()

        # Keyboard Navigation
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and self.current_page > 0:
                self.current_page -= 1
            elif event.key == pygame.K_RIGHT and self.current_page < max_pages - 1:
                self.current_page += 1

        # Mouse Drag / Touch Swipe Navigation
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.drag_start_x = event.pos[0]
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.drag_start_x is not None:
                dx = event.pos[0] - self.drag_start_x
                self.drag_start_x = None
                swipe_threshold = 40
                
                if dx < -swipe_threshold and self.current_page < max_pages - 1:
                    self.current_page += 1
                elif dx > swipe_threshold and self.current_page > 0:
                    self.current_page -= 1

    def draw(self, surface):
        screen_surf = pygame.Surface((self.width, self.height))
        screen_surf.fill((0, 0, 0)) # Pure black classic wallpaper background

        # Status Bar
        pygame.draw.rect(screen_surf, (15, 15, 15), (0, 0, self.width, 20))
        
        # Real Live Clock
        current_time_str = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        font_sm = pygame.font.SysFont(None, 15)
        time_lbl = font_sm.render(current_time_str, True, (255, 255, 255))
        screen_surf.blit(time_lbl, (self.width // 2 - time_lbl.get_width() // 2, 3))

        # App Grid Layout Dimensions (Shifted UP)
        icon_size, margin_x, margin_y = 56, 24, 28
        grid_width = (4 * icon_size) + (3 * margin_x)
        start_x = (self.width - grid_width) // 2
        start_y = 30  # Moved up from 50 to 30

        app_list = list(self.apps_metadata.keys())
        max_pages = self.total_pages()
        
        # Ensure page index stays in bounds
        if self.current_page >= max_pages:
            self.current_page = max(0, max_pages - 1)

        # Get apps for the current page
        page_start = self.current_page * self.apps_per_page
        page_apps = app_list[page_start:page_start + self.apps_per_page]

        for i, app_name in enumerate(page_apps):
            col, row = i % 4, i // 4
            x, y = start_x + col * (icon_size + margin_x), start_y + row * (icon_size + margin_y)
            
            app_data = self.apps_metadata[app_name]
            icon_surf = app_data.get('icon')

            if icon_surf:
                screen_surf.blit(icon_surf, (x, y))
            else:
                icon_rect = pygame.Rect(x, y, icon_size, icon_size)
                pygame.draw.rect(screen_surf, (240, 240, 240), icon_rect, border_radius=12)
                
                gloss_surf = pygame.Surface((icon_size, icon_size // 2), pygame.SRCALPHA)
                pygame.draw.rect(gloss_surf, (255, 255, 255, 80), (0, 0, icon_size, icon_size // 2), border_top_left_radius=12, border_top_right_radius=12)
                screen_surf.blit(gloss_surf, (x, y))
            
            font_lbl = pygame.font.SysFont(None, 14)
            display_label = app_name if len(app_name) <= 10 else app_name[:8] + ".."
            lbl = font_lbl.render(display_label, True, (255, 255, 255))
            screen_surf.blit(lbl, (x + (icon_size - lbl.get_width()) // 2, y + icon_size + 4))

        if not app_list:
            screen_surf.blit(font_sm.render("Drop .ipa files into /apps", True, (200, 200, 200)), (30, 200))

        # Pagination Bar (Dots & Arrows above the Dock)
        page_bar_y = self.height - 88
        if max_pages > 1:
            dot_spacing = 10
            total_dots_w = (max_pages * 6) + ((max_pages - 1) * dot_spacing)
            start_dot_x = (self.width - total_dots_w) // 2

            for p in range(max_pages):
                dot_x = start_dot_x + p * (6 + dot_spacing)
                color = (255, 255, 255) if p == self.current_page else (100, 100, 100)
                pygame.draw.circle(screen_surf, color, (dot_x + 3, page_bar_y), 3)

            # Left Arrow
            font_nav = pygame.font.SysFont(None, 18)
            if self.current_page > 0:
                prev_lbl = font_nav.render("<", True, (200, 200, 200))
                screen_surf.blit(prev_lbl, (15, page_bar_y - 6))

            # Right Arrow
            if self.current_page < max_pages - 1:
                next_lbl = font_nav.render(">", True, (200, 200, 200))
                screen_surf.blit(next_lbl, (self.width - 22, page_bar_y - 6))

        # Classic iOS Dock
        dock_rect = pygame.Rect(10, self.height - 75, self.width - 20, 65)
        pygame.draw.rect(screen_surf, (20, 20, 20), dock_rect, border_radius=16)

        scaled = pygame.transform.scale(screen_surf, (self.width * 2, self.height * 2))
        surface.blit(scaled, (0, 0))

    def handle_click(self, pos):
        vx, vy = pos[0] // 2, pos[1] // 2
        max_pages = self.total_pages()
        page_bar_y = self.height - 88

        # Left/Right Button Click Targets
        if max_pages > 1 and abs(vy - page_bar_y) <= 12:
            if vx < 30 and self.current_page > 0:
                self.current_page -= 1
                return None
            elif vx > self.width - 30 and self.current_page < max_pages - 1:
                self.current_page += 1
                return None

        # App Icon Clicks for Active Page
        icon_size, margin_x, margin_y = 56, 24, 28
        grid_width = (4 * icon_size) + (3 * margin_x)
        start_x = (self.width - grid_width) // 2
        start_y = 30

        app_list = list(self.apps_metadata.keys())
        page_start = self.current_page * self.apps_per_page
        page_apps = app_list[page_start:page_start + self.apps_per_page]

        for i, app_name in enumerate(page_apps):
            col, row = i % 4, i // 4
            x, y = start_x + col * (icon_size + margin_x), start_y + row * (icon_size + margin_y)
            if pygame.Rect(x, y, icon_size, icon_size).collidepoint(vx, vy):
                return self.apps_metadata[app_name]

        return None