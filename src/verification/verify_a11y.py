"""Accessibility verification script for Gas Density and Custom Unit inputs."""

import logging

from playwright.sync_api import expect, sync_playwright

logger = logging.getLogger(__name__)


def verify_a11y() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8080")

        logger.info("Checking Gas Density input...")
        # Select Heating Value category
        page.select_option("#category", "heating_value")

        # Check gasDensity input
        gas_density = page.locator("#gasDensity")
        expect(gas_density).to_be_visible()
        described_by = gas_density.get_attribute("aria-describedby")
        logger.info(f"gasDensity aria-describedby: {described_by}")
        if (
            described_by
            and "gasDensityHint" in described_by
            and "heatingValHint" in described_by
        ):
            logger.info("PASS: gasDensity has correct describedby")
        else:
            logger.error("FAIL: gasDensity missing describedby")

        logger.info("\nChecking Custom Unit inputs...")
        # Open Custom Units modal
        page.click("#customUnitsButton")

        # Check conversionFactor input
        conv_factor = page.locator("#conversionFactor")
        expect(conv_factor).to_be_visible()
        described_by = conv_factor.get_attribute("aria-describedby")
        logger.info(f"conversionFactor aria-describedby: {described_by}")
        if described_by and "conversionFactorHint" in described_by:
            logger.info("PASS: conversionFactor has correct describedby")
        else:
            logger.error("FAIL: conversionFactor missing describedby")

        # Check customAliases input
        aliases = page.locator("#customAliases")
        expect(aliases).to_be_visible()
        described_by = aliases.get_attribute("aria-describedby")
        logger.info(f"customAliases aria-describedby: {described_by}")
        if described_by and "customAliasesHint" in described_by:
            logger.info("PASS: customAliases has correct describedby")
        else:
            logger.error("FAIL: customAliases missing describedby")

        page.screenshot(path="verification/verification.png")
        browser.close()


if __name__ == "__main__":
    verify_a11y()
