#!/usr/bin/env python3
"""
Exact DAPA Scraper
Accurately collects terms from <ul id="dicti_list_ul">
Handles pagination using goPaging() function
"""

import requests
from bs4 import BeautifulSoup
import time
import json


def scrape_dapa_terms():
    """Accurate scraping of DAPA defense dictionary"""

    print("🎯 Starting accurate DAPA defense dictionary scraping")
    print("=" * 50)

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.8',
        'Connection': 'keep-alive',
        'Referer': 'http://www.dapa.go.kr/dapa/wd/wdcry/list.do'
    })

    base_url = "http://www.dapa.go.kr/dapa/wd/wdcry/list.do"
    all_terms = []

    # Start with first page and get total page count
    print("📄 Accessing first page...")

    try:
        response = session.get(base_url, params={'menuId': '326'}, timeout=15)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract terms from first page
        first_page_terms = extract_terms_from_page(soup, 1)
        all_terms.extend(first_page_terms)

        # Get maximum page number
        max_page = get_max_page(soup)
        print(f"📊 Total pages: {max_page}")

        # Process all pages (no limit)
        pages_to_process = max_page  # All 899 pages!
        print(f"📋 Pages to process: 1~{pages_to_process} (complete!)")

        # Process remaining pages
        for page in range(2, pages_to_process + 1):
            print(f"📄 Processing page {page:3d}/{pages_to_process}... ({100 * page // pages_to_process:2d}%)")

            # POST request to navigate pages (mimicking goPaging function)
            page_data = {
                'currPage': str(page),
                'dicti_sel1': '',
                'firstWord': '',
                'minSn': '0',
                'dicti_sel2': '',
                'dicti_sel3': '',
                'menuId': '326',
                'firstWordType': '',
                'maxSn': '20',
                'searchValue': '',
                'searchType3': ''
            }

            try:
                response = session.post(base_url, data=page_data, timeout=15)
                response.encoding = 'utf-8'
                page_soup = BeautifulSoup(response.content, 'html.parser')

                page_terms = extract_terms_from_page(page_soup, page)
                all_terms.extend(page_terms)

                # Progress update (every 50 pages)
                if page % 50 == 0:
                    print(
                        f"  📊 Progress: {page}/{pages_to_process} ({100 * page // pages_to_process}%) | Total terms: {len(all_terms)}")

            except Exception as e:
                print(f"  ❌ Page {page} failed: {e}")
                continue

            # Server load balancing (prevent blocking)
            time.sleep(0.8)

        # Remove duplicates while preserving order
        unique_terms = list(dict.fromkeys(all_terms))

        print(f"✅ Scraping completed!")
        print(f"📊 Total collected terms: {len(unique_terms)} (before dedup: {len(all_terms)})")
        print(f"📄 Processed pages: {pages_to_process}")
        print(f"⚡ Average terms per page: {len(unique_terms) / pages_to_process:.1f}")

        return unique_terms

    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return []


def extract_terms_from_page(soup, page_num):
    """Extract exact terms from page"""
    terms = []

    # Exact target: <ul id="dicti_list_ul">
    dicti_list_ul = soup.find('ul', id='dicti_list_ul')

    if dicti_list_ul:
        # Extract text from <a> tags within each <li>
        li_elements = dicti_list_ul.find_all('li')

        for li in li_elements:
            a_tag = li.find('a')
            if a_tag:
                term = a_tag.get_text(strip=True)
                if term:
                    terms.append(term)

        print(f"  ✅ Page {page_num}: {len(terms)} terms")

        # Preview first few terms (first page only)
        if page_num == 1 and terms:
            print("  📝 Sample terms:")
            for i, term in enumerate(terms[:5], 1):
                print(f"    {i}. {term}")
    else:
        print(f"  ❌ Page {page_num}: dicti_list_ul not found")

    return terms


def get_max_page(soup):
    """Extract maximum page number"""
    try:
        # Extract page number from last page button
        last_button = soup.find('a', {
            'onclick': lambda x: x and 'goPaging' in x and '마지막' in soup.find('a', {'onclick': x}).get('title', '')})

        if last_button:
            onclick = last_button.get('onclick', '')
            # Extract 899 from goPaging(899)
            import re
            match = re.search(r'goPaging\((\d+)\)', onclick)
            if match:
                return int(match.group(1))

        # Alternative: check page numbers in dicti_pag_btn1
        paging_area = soup.find('p', class_='dicti_pag_btn1')
        if paging_area:
            page_links = paging_area.find_all('a')
            page_numbers = []
            for link in page_links:
                onclick = link.get('onclick', '')
                match = re.search(r'goPaging\((\d+)\)', onclick)
                if match:
                    page_numbers.append(int(match.group(1)))

            if page_numbers:
                # Estimate from maximum visible page number
                return max(page_numbers) * 20  # estimation

        # Default value
        return 50

    except Exception as e:
        print(f"Failed to extract page count: {e}")
        return 50


def save_results(terms):
    """Save results to files"""
    if not terms:
        print("❌ No terms to save.")
        return

    # Save to JSON
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_terms': len(terms),
        'source': 'DAPA Defense Dictionary (exact extraction)',
        'terms': terms
    }

    with open('dapa_exact_terms.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Save to text file (without numbering)
    with open('dapa_exact_terms.txt', 'w', encoding='utf-8') as f:
        for term in terms:
            f.write(f"{term}\n")

    print(f"\n💾 Results saved:")
    print(f"  📄 dapa_exact_terms.json")
    print(f"  📝 dapa_exact_terms.txt")


def main():
    """Main execution function"""
    start_time = time.time()

    # Execute scraping
    terms = scrape_dapa_terms()

    end_time = time.time()

    # Process results
    if terms:
        print(f"\n🎉 Success! Collected {len(terms)} exact terms!")
        print(f"⏱️ Execution time: {end_time - start_time:.1f} seconds")

        # Display sample results
        print(f"\n📝 Sample collected terms:")
        for i, term in enumerate(terms[:20], 1):
            print(f"  {i:2d}. {term}")

        if len(terms) > 20:
            print(f"      ... and {len(terms) - 20} more")

        # Save results
        save_results(terms)

    else:
        print("❌ Term collection failed")

    print("=" * 50)


if __name__ == "__main__":
    main()