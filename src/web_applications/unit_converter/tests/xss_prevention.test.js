describe('XSS Prevention', () => {
  // The function to be added to app.js
  function escapeHtml(unsafe) {
    if (unsafe === undefined || unsafe === null) return '';
    return String(unsafe)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  describe('escapeHtml', () => {
    test('escapes basic HTML characters', () => {
      expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
      expect(escapeHtml('"quoted"')).toBe('&quot;quoted&quot;');
      expect(escapeHtml('Start & End')).toBe('Start &amp; End');
    });

    test('escapes XSS payloads', () => {
      const payload = '<img src=x onerror=alert(1)>';
      expect(escapeHtml(payload)).toBe('&lt;img src=x onerror=alert(1)&gt;');
    });

    test('handles non-string inputs', () => {
      expect(escapeHtml(123)).toBe('123');
      expect(escapeHtml(null)).toBe('');
      expect(escapeHtml(undefined)).toBe('');
    });
  });

  describe('Rendering Simulation', () => {
    test('renderCustomUnitsList simulation is safe with escaping', () => {
      const maliciousUnit = '<img src=x onerror=alert(1)>';
      const category = 'length';

      // Vulnerable version logic simulation
      const vulnerableHtml = `
            <div class="custom-unit-item">
              <span class="custom-unit-name">${maliciousUnit}</span>
            </div>
      `;
      expect(vulnerableHtml).toContain('<img src=x onerror=alert(1)>');

      // Secured version logic simulation
      const securedHtml = `
            <div class="custom-unit-item">
              <span class="custom-unit-name">${escapeHtml(maliciousUnit)}</span>
            </div>
      `;
      expect(securedHtml).not.toContain('<img src=x onerror=alert(1)>');
      expect(securedHtml).toContain('&lt;img src=x onerror=alert(1)&gt;');
    });

    test('renderHistory simulation is safe with escaping', () => {
      const item = {
        fromValue: 1,
        fromUnit: '<script>alert(1)</script>',
        toValue: 100,
        toUnit: 'cm',
        category: '<img src=x onerror=alert(1)>' // Malicious category
      };

      const getCategoryLabel = cat => cat; // Mock behavior

      // Vulnerable logic simulation
      const vulnerableHtml = `
        <div class="recent-item">
          <div class="recent-item-text">
            ${item.fromValue} ${escapeHtml(item.fromUnit)} = ${item.toValue} ${escapeHtml(
              item.toUnit
            )}
          </div>
          <div class="recent-item-time">${getCategoryLabel(item.category)} • just now</div>
        </div>
        `;
      expect(vulnerableHtml).toContain('<img src=x onerror=alert(1)>');

      // Secured logic simulation
      const securedHtml = `
        <div class="recent-item">
          <div class="recent-item-text">
            ${item.fromValue} ${escapeHtml(item.fromUnit)} = ${item.toValue} ${escapeHtml(
              item.toUnit
            )}
          </div>
          <div class="recent-item-time">${escapeHtml(
            getCategoryLabel(item.category)
          )} • just now</div>
        </div>
        `;

      expect(securedHtml).not.toContain('<script>');
      expect(securedHtml).toContain('&lt;script&gt;');
      expect(securedHtml).not.toContain('<img');
      expect(securedHtml).toContain('&lt;img src=x onerror=alert(1)&gt;');
    });

    test('renderCustomUnitsList category is safe with escaping', () => {
      const category = '<img src=x onerror=alert(1)>';
      const getCategoryLabel = cat => cat;

      // Vulnerable
      const vulnerableHtml = `<h4 class="custom-category-title">${getCategoryLabel(category)}</h4>`;
      expect(vulnerableHtml).toContain('<img');

      // Secure
      const securedHtml = `<h4 class="custom-category-title">${escapeHtml(
        getCategoryLabel(category)
      )}</h4>`;
      expect(securedHtml).toContain('&lt;img');
    });
  });
});
