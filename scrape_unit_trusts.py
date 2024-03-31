import asyncio
import json
import os
from playwright.async_api import async_playwright
from tabulate import tabulate


async def save_bytes_to_file(data, filename):
    with open(filename, "wb") as f:
        f.write(data)


async def scrape_pb_unit_trusts():
    # Scrape unit trusts from the internet
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        page = await browser.new_page()
        await page.goto("https://www.publicmutual.com.my/Our-Products/Fund-Explorer")
        await page.get_by_text("Search").and_(
            page.locator(".btn.btn-primary.button-color-red")
        ).click()
        async with page.expect_response(
            lambda response: response.url
            == "https://www.publicmutual.com.my/Our-Products/Fund-Explorer"
            and response.status == 200
        ):
            print("Waiting for response")
        print("Response received")
        await page.wait_for_timeout(1000)
        # Scrape the unit trusts from the table
        table_rows = (
            await page.locator("tr").and_(page.locator(":not(.gvw-header-light)")).all()
        )
        print(f"Found {len(table_rows)} unit trusts")

        # Parse the table rows
        unit_trusts = []
        for row in table_rows:
            columns = await row.locator("td").all()
            phs_file_name = f"pb_phs/phs_{await columns[0].inner_text()}.pdf"

            unit_trust = {
                "fund_name": await columns[0].inner_text(),
                "shariah_compliant": await columns[1].inner_text(),
                "distribution_policy": await columns[2].inner_text(),
                "risk_level": await columns[3].inner_text(),
                "phs_en": phs_file_name.split("/")[1],
                # "prospectus_en": await columns[7].inner_text(),
            }
            unit_trusts.append(unit_trust)

        # Save the unit trusts to a JSON file
        with open("unit_trusts.json", "w") as f:
            json.dump(unit_trusts, f, indent=4)

        downloaded_phs_funds = [
            phs.split("_")[1].split(".")[0] for phs in os.listdir("pb_phs")
        ]
        print(str(len(downloaded_phs_funds)) + " PHS files already downloaded")

        for row in table_rows:
            columns = await row.locator("td").all()
            phs_file_name = f"pb_phs/phs_{await columns[0].inner_text()}.pdf"
            # Download PHS only if it has not been downloaded before
            if await columns[0].inner_text() in downloaded_phs_funds:
                print(f"PHS for {await columns[0].inner_text()} already downloaded")
                continue
            # Download PHS
            async with page.expect_popup() as popup_info:
                await columns[4].locator("a").click()
            popup = await popup_info.value
            print(f"Downloading PHS for {await columns[0].inner_text()}")
            response = await popup.context.request.get(popup.url)
            pdfBuffer = await response.body()
            await save_bytes_to_file(pdfBuffer, phs_file_name)
            await popup.close()
            print(f"Downloaded PHS for {await columns[0].inner_text()}")


async def parse_pb_unit_trusts():
    print("E")


async def main():
    # await scrape_pb_unit_trusts()
    await parse_pb_unit_trusts()


if __name__ == "__main__":
    asyncio.run(main())
