# Deployment Guide

## Quick Start - GitHub Pages (Recommended for iOS)

### 1. Prepare Repository

```bash
cd /home/user/Gasification_Model
git add unit-converter-app/
git commit -m "Add standalone unit converter PWA"
git push -u origin claude/standalone-unit-converter-app-01BeDFaJRMujwcENzVBSPZQ5
```

### 2. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under "Source", select your branch: `claude/standalone-unit-converter-app-01BeDFaJRMujwcENzVBSPZQ5`
4. Under "Folder", select `/ (root)` or select the branch root
5. Click **Save**

Your app will be available at:

```
https://D-sorganization.github.io/Gasification_Model/unit-converter-app/
```

### 3. Update Manifest (if needed)

If your app is not at the root path, update `manifest.json`:

```json
{
  "start_url": "/Gasification_Model/unit-converter-app/",
  ...
}
```

And update `service-worker.js` cache URLs:

```javascript
const urlsToCache = [
  '/Gasification_Model/unit-converter-app/',
  '/Gasification_Model/unit-converter-app/index.html'
  // ... etc
];
```

### 4. Install on iOS

Once deployed:

1. Open **Safari** on your iPhone
2. Go to `https://D-sorganization.github.io/Gasification_Model/unit-converter-app/`
3. Tap the **Share** button (□ with ↑)
4. Scroll and tap **"Add to Home Screen"**
5. Tap **"Add"**
6. App appears on your home screen like a native app!

---

## Alternative Deployment Options

### Option 1: Netlify (Free, Automatic HTTPS)

1. Create account at [netlify.com](https://netlify.com)
2. Drag and drop the `unit-converter-app` folder
3. Get instant URL: `https://your-app-name.netlify.app`
4. Custom domain supported

### Option 2: Vercel (Free, Optimized)

```bash
npm install -g vercel
cd unit-converter-app
vercel
```

### Option 3: Your Own Server

Requirements:

- HTTPS (required for PWA)
- Static file server

Example with Python:

```bash
cd unit-converter-app
python -m http.server 8000
```

With nginx:

```nginx
server {
    listen 443 ssl;
    server_name converter.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    root /path/to/unit-converter-app;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## Icon Generation

The app includes `icon.svg`. To generate PNG icons:

### Using ImageMagick:

```bash
# Install ImageMagick
brew install imagemagick  # macOS
# or
sudo apt-get install imagemagick  # Linux

# Generate icons
convert icon.svg -resize 192x192 icon-192.png
convert icon.svg -resize 512x512 icon-512.png
```

### Using Online Tool:

1. Go to [https://realfavicongenerator.net/](https://realfavicongenerator.net/)
2. Upload `icon.svg`
3. Download all sizes
4. Place in `unit-converter-app/` folder

### Using Figma/Sketch:

1. Import `icon.svg`
2. Export as PNG at 192x192 and 512x512
3. Save as `icon-192.png` and `icon-512.png`

---

## Testing Before Deployment

### Local Testing

1. Start local server:

   ```bash
   cd /home/user/Gasification_Model
   python -m http.server 8000
   ```

2. Open browser:

   ```
   http://localhost:8000/unit-converter-app/
   ```

3. Test features:
   - [ ] Unit conversions work
   - [ ] Theme toggle works
   - [ ] History saves and loads
   - [ ] Swap button works
   - [ ] All categories have units
   - [ ] Responsive on mobile

### PWA Testing

Use Chrome DevTools:

1. Open DevTools (F12)
2. Go to **Application** tab
3. Check:
   - [ ] Manifest loads correctly
   - [ ] Service Worker registers
   - [ ] Cache populated
   - [ ] Icons display

### iOS Testing

Use Safari on iPhone:

1. Load the page
2. Check:
   - [ ] Layout fits screen (no horizontal scroll)
   - [ ] Inputs work (number pad appears)
   - [ ] Install prompt shows (if first visit)
   - [ ] Touch targets are large enough
   - [ ] Notch/safe areas respected

---

## Troubleshooting

### Service Worker Not Registering

- Check console for errors
- Ensure HTTPS (required for SW)
- Clear browser cache
- Update cache version in `service-worker.js`

### Icons Not Showing

- Generate PNG icons from SVG
- Check manifest paths are correct
- Verify icons are in same folder as manifest

### App Not Installing on iOS

- Must use Safari (not Chrome/Firefox on iOS)
- Needs HTTPS
- Manifest must be valid JSON
- Icons must exist at specified paths

### Offline Not Working

- Service Worker must register successfully
- All files must be cached
- Check DevTools → Application → Cache Storage
- Update URLs in service-worker.js to match deployment

### Conversions Failing

- Check browser console for errors
- Verify units are spelled correctly
- Ensure category has loaded units
- Check that values are valid numbers

---

## Updating the App

1. Make changes to files
2. Update cache version in `service-worker.js`:
   ```javascript
   const CACHE_NAME = 'unit-converter-v1.0.1'; // Increment
   ```
3. Commit and push changes
4. Users will get update next time they open app
5. Or force refresh: Settings → Safari → Clear History and Website Data

---

## Performance Optimization

### Already Implemented:

- ✓ Minimal dependencies (no frameworks)
- ✓ Service Worker caching
- ✓ Local storage for history
- ✓ Efficient conversion algorithms
- ✓ CSS minification ready

### Optional Enhancements:

- Minify JavaScript: `terser app.js -o app.min.js`
- Minify CSS: `cssnano styles.css styles.min.css`
- Compress images: Use WebP for icons
- Enable Brotli compression on server

---

## Security Considerations

- App runs entirely client-side (no server data)
- No external dependencies or CDNs
- No analytics or tracking
- Conversion history stored locally only
- HTTPS required for PWA features

---

## Browser Compatibility

| Browser          | Support               |
| ---------------- | --------------------- |
| Safari iOS 11.3+ | ✓ Full (PWA install)  |
| Chrome Android   | ✓ Full (PWA install)  |
| Safari macOS     | ✓ Works (no install)  |
| Chrome Desktop   | ✓ Full (PWA install)  |
| Firefox          | ✓ Works (limited PWA) |
| Edge             | ✓ Full (PWA install)  |

---

## Support

For issues or questions:

1. Check console for errors
2. Verify all files are present
3. Test in different browser
4. Check GitHub repository for updates
