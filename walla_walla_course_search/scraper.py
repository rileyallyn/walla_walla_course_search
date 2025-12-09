"""
Recursive web scraper for Walla Walla University Computer Science courses.
Extracts course information from the main page and all linked course pages.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
from typing import Set, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourseScraper:
    def __init__(self, base_url: str, delay: float = 1.0):
        """
        Initialize the scraper.
        
        Args:
            base_url: The base URL of the course catalog page
            delay: Delay between requests in seconds (to be respectful)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        user_agent = (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36'
        )
        self.session.headers.update({'User-Agent': user_agent})
        self.visited_urls: Set[str] = set()
        self.courses: List[Dict] = []

    def _normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates."""
        parsed = urlparse(url)
        # Remove fragment and normalize path
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized

    def _get_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page."""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _is_course_catalog_url(self, url: str) -> bool:
        """Check if URL is part of the course catalog domain."""
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        return parsed.netloc == base_parsed.netloc

    def _extract_course_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract all course detail page links from the main page."""
        links = []
        if not soup:
            return links

        # Find all links that might be course pages
        # Course links might be in various formats
        ul = soup.find('ul', class_='sc-child-item-links')
        if ul:
            for link in ul.find_all('a', href=True):
                href = link.get('href')
                if not href:
                    continue

                full_url = urljoin(self.base_url, href)
                normalized = self._normalize_url(full_url)

                # Look for course links - might contain course codes or numbers
                # Also check for links that go deeper into the catalog
                # structure
                if self._is_course_catalog_url(normalized):
                    # Include links to course detail pages or sub-pages
                    # Exclude external links, anchors, and navigation links
                    if not any(exclude in normalized.lower() for exclude in [
                        '#', 'javascript:', 'mailto:', 'tel:'
                    ]):
                        links.append(normalized)

        return links

    def _extract_course_info(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract course information from a course detail page."""
        if not soup:
            return None

        course_info = {
            'url': url,
            'title': '',
            'code': '',
            'description': '',
            'credits': '',
            'prerequisites': '',
            'corequisites': '',
            'distribution': '',
            'full_text': ''
        }
        
        # Extract full text for processing
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        course_info['full_text'] = ' '.join(text.split())
        
        # Look for the main content div
        main_div = soup.find('div', id='main')
        if not main_div:
            # Fallback: try to find course title (usually in h1 or h2)
            title_elem = soup.find('h1') or soup.find('h2') or soup.find('h3')
            if title_elem:
                course_info['title'] = title_elem.get_text(strip=True)
            
            # Try to extract course code using regex as fallback
            import re
            code_pattern = r'\b(CPTR|CIS|CYBS|GDEV)\s*[-\s]?\s*(\d{3})\b'
            code_match = re.search(
                code_pattern, course_info['full_text'], re.IGNORECASE
            )
            if code_match:
                code = f"{code_match.group(1).upper()} {code_match.group(2)}"
                course_info['code'] = code
            
            # Try to find course description
            desc_selectors = [
                'div.course-description',
                'div.description',
                'div.desc',
                'p.description',
                'div.course-info',
                'div.content'
            ]
            
            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    course_info['description'] = desc_elem.get_text(strip=True)
                    break
            
            return course_info
        
        # Extract course code from span in h1
        h1 = main_div.find('h1')
        if h1:
            code_span = h1.find('span')
            if code_span:
                course_info['code'] = code_span.get_text(strip=True)
                # Extract title (everything in h1 except the span)
                title_text = h1.get_text()
                if code_span.get_text() in title_text:
                    title_text = title_text.replace(code_span.get_text(), '', 1)
                course_info['title'] = title_text.strip()
            else:
                # Fallback: extract from h1 text
                h1_text = h1.get_text(strip=True)
                course_info['title'] = h1_text
                # Try to extract code from title
                import re
                code_pattern = r'\b(CPTR|CIS|CYBS|GDEV)\s*[-\s]?\s*(\d{3})\b'
                code_match = re.search(code_pattern, h1_text, re.IGNORECASE)
                if code_match:
                    course_info['code'] = f"{code_match.group(1).upper()} {code_match.group(2)}"
        
        # Extract description from div.desc
        desc_div = main_div.find('div', class_='desc')
        if desc_div:
            course_info['description'] = desc_div.get_text(strip=True)
        
        # Extract credits from div#credits
        credits_div = main_div.find('div', id='credits')
        if credits_div:
            # Remove the h3 "Credits" text and get the number
            credits_text = credits_div.get_text(strip=True)
            # Remove "Credits" header text
            credits_text = credits_text.replace('Credits', '').strip()
            course_info['credits'] = credits_text
        
        # Extract prerequisites from div.sc_prereqs
        prereqs_div = main_div.find('div', class_='sc_prereqs')
        if prereqs_div:
            # Get all prerequisite links
            prereq_links = prereqs_div.find_all('a', class_='sc-courselink')
            if prereq_links:
                prereq_codes = [link.get_text(strip=True) for link in prereq_links]
                course_info['prerequisites'] = ', '.join(prereq_codes)
            else:
                # If no links, get the text (excluding the h3 header)
                prereq_text = prereqs_div.get_text(strip=True)
                # Remove "Prerequisite" header
                prereq_text = prereq_text.replace('Prerequisite', '').strip()
                if prereq_text:
                    course_info['prerequisites'] = prereq_text
        
        # Extract corequisites from div.sc_coreqs
        coreqs_div = main_div.find('div', class_='sc_coreqs')
        if coreqs_div:
            coreq_links = coreqs_div.find_all('a', class_='sc-courselink')
            if coreq_links:
                coreq_codes = [link.get_text(strip=True) for link in coreq_links]
                course_info['corequisites'] = ', '.join(coreq_codes)
            else:
                # If no links, get the text (excluding the h3 header)
                coreq_text = coreqs_div.get_text(strip=True)
                # Remove "Corequisite" header if present
                coreq_text = coreq_text.replace('Corequisite', '').strip()
                if coreq_text:
                    course_info['corequisites'] = coreq_text
        
        # Extract distribution from div#distribution
        dist_div = main_div.find('div', id='distribution')
        if dist_div:
            # Remove the h3 "Distribution" text and get the value
            dist_text = dist_div.get_text(strip=True)
            dist_text = dist_text.replace('Distribution', '').strip()
            course_info['distribution'] = dist_text

        return course_info

    def scrape_main_page(self) -> List[Dict]:
        """Scrape the main course catalog page and extract all course links."""
        logger.info(f"Scraping main page: {self.base_url}")
        soup = self._get_page(self.base_url)
        
        if not soup:
            logger.error("Failed to fetch main page")
            return []
        
        # First, extract course information from the main page itself
        main_course_info = self._extract_course_info(soup, self.base_url)
        if main_course_info:
            self.courses.append(main_course_info)
        
        # Find all course links
        course_links = self._extract_course_links(soup)
        logger.info(f"Found {len(course_links)} potential course links")

        # Also check for links organized by course level (100, 200, 300, 400)
        # The page might have links to sub-pages for each level
        level_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text(strip=True).lower()
            # Look for links that mention course levels
            if any(level in text for level in ['100', '200', '300', '400']):
                full_url = urljoin(self.base_url, href)
                normalized = self._normalize_url(full_url)
                if self._is_course_catalog_url(normalized):
                    level_links.append(normalized)
        
        # Combine all links to visit
        all_links = list(set(course_links + level_links))

        return all_links

    def scrape_recursive(self, max_depth: int = 3, current_depth: int = 0):
        """
        Recursively scrape course pages.
        
        Args:
            max_depth: Maximum depth to crawl
            current_depth: Current crawling depth
        """
        if current_depth >= max_depth:
            return
        
        if current_depth == 0:
            # Start with main page
            links_to_visit = self.scrape_main_page()
        else:
            # This would be called recursively for sub-pages
            # For now, we'll handle it iteratively
            return
        
        # Visit each link found
        for link in links_to_visit:
            normalized = self._normalize_url(link)

            if normalized in self.visited_urls:
                continue

            self.visited_urls.add(normalized)
            logger.info(
                f"Scraping: {normalized} (depth: {current_depth})"
            )

            soup = self._get_page(normalized)
            if soup:
                course_info = self._extract_course_info(soup, normalized)
                if course_info and course_info.get('code'):
                    self.courses.append(course_info)
                    code = course_info.get('code', 'Unknown')
                    logger.info(f"Extracted course: {code}")

                # If this looks like a level page, extract more links
                if current_depth < max_depth - 1:
                    sub_links = self._extract_course_links(soup)
                    for sub_link in sub_links:
                        sub_normalized = self._normalize_url(sub_link)
                        is_visited = sub_normalized not in self.visited_urls
                        is_catalog = self._is_course_catalog_url(
                            sub_normalized
                        )
                        if is_visited and is_catalog:
                            # Recursively scrape
                            self.visited_urls.add(sub_normalized)
                            sub_soup = self._get_page(sub_normalized)
                            if sub_soup:
                                sub_info = self._extract_course_info(
                                    sub_soup, sub_normalized
                                )
                                if sub_info:
                                    self.courses.append(sub_info)
                                    sub_code = sub_info.get('code', 'Unknown')
                                    logger.info(f"Extracted course: {sub_code}")

        logger.info(f"Scraping complete. Found {len(self.courses)} courses.")
    
    def save_courses(self, filename: str = 'courses.json'):
        """Save scraped courses to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.courses, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.courses)} courses to {filename}")

    def get_courses(self) -> List[Dict]:
        """Return the list of scraped courses."""
        return self.courses


if __name__ == '__main__':
    base_url = (
        'https://wallawalla.smartcatalogiq.com/current/'
        'undergraduate-bulletin/courses/cptr-computer-science'
    )
    scraper = CourseScraper(base_url, delay=1.0)
    scraper.scrape_recursive(max_depth=3)
    scraper.save_courses('courses.json')
    print(f"\nScraped {len(scraper.get_courses())} courses")

