#!/usr/bin/env python3
"""Query arXiv API and print structured results."""
import sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}

def search(query, max_results=8):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            break
        except Exception as e:
            print(f"  retry {attempt}: {e}", file=sys.stderr)
            time.sleep(3)
    else:
        return []
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.strip()
        title = " ".join(e.find("a:title", NS).text.split())
        summary = " ".join(e.find("a:summary", NS).text.split())
        published = e.find("a:published", NS).text[:10]
        authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        out.append({"id": aid, "title": title, "summary": summary,
                    "published": published, "authors": authors})
    return out

if __name__ == "__main__":
    q = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    res = search(q, n)
    print(f"=== QUERY: {q}  ({len(res)} results) ===\n")
    for r in res:
        print(f"[{r['published']}] {r['title']}")
        print(f"  {r['id']}")
        print(f"  Authors: {', '.join(r['authors'][:6])}")
        print(f"  {r['summary'][:600]}")
        print()
