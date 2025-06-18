"""
Web scraper with error handling issues for testing the SWE agent.
"""

import time
from urllib.parse import urlparse

import requests


class WebScraper:
    def __init__(self, base_url: str, delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.scraped_urls = []

    def fetch_page(self, url: str) -> str:
        """Fetch a web page."""
        # Bug: No error handling for HTTP errors or network issues
        response = self.session.get(url)
        return response.text

    def fetch_multiple_pages(self, urls: list[str]) -> dict[str, str]:
        """Fetch multiple web pages."""
        results = {}

        for url in urls:
            # Bug: No delay between requests (can overwhelm server)
            # Bug: No error handling - one failed request breaks everything
            content = self.fetch_page(url)
            results[url] = content
            self.scraped_urls.append(url)

        return results

    def extract_links(self, html_content: str, base_url: str) -> list[str]:
        """Extract all links from HTML content."""
        # Bug: Very naive implementation that doesn't handle edge cases
        links = []

        # Bug: This regex-like approach is fragile and error-prone
        import re

        href_pattern = r'href=["\']([^"\']*)["\']'
        matches = re.findall(href_pattern, html_content)

        for match in matches:
            # Bug: No validation of URLs or handling of relative URLs properly
            if match.startswith("http"):
                links.append(match)
            else:
                # Bug: Doesn't handle cases where base_url doesn't end with /
                full_url = base_url + match
                links.append(full_url)

        return links

    def scrape_with_retries(self, url: str, max_retries: int = 3) -> str | None:
        """Scrape a URL with retry logic."""
        retries = 0

        while retries < max_retries:
            try:
                # Bug: No exponential backoff - always waits the same amount
                time.sleep(self.delay)
                response = self.session.get(url)

                # Bug: Only checks for 200, ignores other success codes like 201, 202
                if response.status_code == 200:
                    return response.text
                else:
                    retries += 1

            except Exception:
                # Bug: Doesn't log the specific error or URL that failed
                retries += 1

        # Bug: Returns None instead of raising an exception or logging failure
        return None

    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL."""
        # Bug: No error handling for network issues or invalid URLs
        response = self.session.get(url)

        # Bug: No check for content-type or file size limits
        with open(filename, "wb") as file:
            # Bug: Downloads entire file into memory first (bad for large files)
            file.write(response.content)

        return True

    def parse_robots_txt(self, domain: str) -> dict[str, list[str]]:
        """Parse robots.txt file."""
        robots_url = f"http://{domain}/robots.txt"

        # Bug: Forces HTTP instead of checking HTTPS first
        # Bug: No error handling if robots.txt doesn't exist
        content = self.fetch_page(robots_url)

        rules = {"allow": [], "disallow": []}

        for line in content.split("\n"):
            line = line.strip()
            # Bug: Case-sensitive matching (should be case-insensitive)
            if line.startswith("Allow:"):
                # Bug: Doesn't strip whitespace after colon
                path = line.split(":", 1)[1]
                rules["allow"].append(path)
            elif line.startswith("Disallow:"):
                path = line.split(":", 1)[1]
                rules["disallow"].append(path)

        return rules

    def is_url_allowed(self, url: str, robots_rules: dict[str, list[str]]) -> bool:
        """Check if URL is allowed by robots.txt."""
        parsed_url = urlparse(url)
        path = parsed_url.path

        # Bug: Logic is backwards - checks disallow before allow
        for disallowed_path in robots_rules.get("disallow", []):
            if path.startswith(disallowed_path):
                return False

        # Bug: If no explicit allow rules, should default to True
        for allowed_path in robots_rules.get("allow", []):
            if path.startswith(allowed_path):
                return True

        return False

    def scrape_sitemap(self, sitemap_url: str) -> list[str]:
        """Extract URLs from a sitemap."""
        # Bug: No error handling for invalid XML or network issues
        content = self.fetch_page(sitemap_url)

        # Bug: Very basic XML parsing that doesn't handle namespaces
        import re

        url_pattern = r"<loc>(.*?)</loc>"
        urls = re.findall(url_pattern, content)

        return urls


def main():
    """Demo function with web scraping bugs."""
    scraper = WebScraper("https://example.com")

    try:
        # Bug: Will fail with network errors or invalid URLs
        content = scraper.fetch_page("https://httpbin.org/html")
        print(f"Fetched {len(content)} characters")

        # Bug: Will fail if any URL in the list is invalid
        multiple_pages = scraper.fetch_multiple_pages(
            [
                "https://httpbin.org/json",
                "https://httpbin.org/xml",
                "https://invalid-url-that-doesnt-exist.com",
            ]
        )
        print(f"Fetched {len(multiple_pages)} pages")

        # Bug: Will fail for domains without robots.txt
        robots = scraper.parse_robots_txt("github.com")
        print(f"Robots.txt rules: {robots}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
