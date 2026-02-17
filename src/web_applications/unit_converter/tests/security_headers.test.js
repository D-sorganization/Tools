const fs = require('fs');
const path = require('path');

describe('Security Headers', () => {
  const htmlPath = path.join(__dirname, '../unit-converter-app/index.html');
  let htmlContent;

  beforeAll(() => {
    htmlContent = fs.readFileSync(htmlPath, 'utf8');
  });

  test('Content-Security-Policy meta tag is present', () => {
    // Check for the presence of the CSP meta tag
    expect(htmlContent).toMatch(/<meta\s+http-equiv=["']Content-Security-Policy["']/i);
  });

  test('CSP contains correct directives', () => {
    // Extract the content attribute of the CSP meta tag
    // Match content attribute with matching quotes
    const regex =
      /<meta\s+http-equiv=["']Content-Security-Policy["']\s+content=(["'])([\s\S]*?)\1/i;
    const cspMatch = htmlContent.match(regex);

    expect(cspMatch).not.toBeNull();
    const cspContent = cspMatch[2]; // Group 2 is the content

    // Check individual directives
    expect(cspContent).toContain("default-src 'self'");
    expect(cspContent).toContain("script-src 'self'");
    expect(cspContent).toContain("style-src 'self'");
    expect(cspContent).not.toContain("'unsafe-inline'");
    expect(cspContent).toContain("img-src 'self' data:");
    expect(cspContent).toContain("object-src 'none'");
    expect(cspContent).toContain("base-uri 'self'");
  });
});
