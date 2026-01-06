from bs4 import BeautifulSoup
import json

def segment_article(html_content):
    soup = BeautifulSoup(html_content, "lxml")

    blocks = []
    block_id = 1
    order = 1
    current_heading_level = 0

    # Define tags we care about, in reading order
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        text = tag.get_text(strip=True)

        if not text:
            continue

        if tag.name.startswith("h"):
            role = "heading"
            level = int(tag.name[1])
            current_heading_level = level

        elif tag.name == "p":
            role = "paragraph"
            level = current_heading_level + 1

        elif tag.name == "li":
            role = "bullet_point"
            level = current_heading_level + 2

        block = {
            "block_id": f"B{block_id}",
            "role": role,
            "hierarchy_level": level,
            "order": order,
            "content": text
        }

        blocks.append(block)
        block_id += 1
        order += 1

    return blocks


if __name__ == "__main__":
    with open("article.html", "r", encoding="utf-8") as f:
        html = f.read()

    result = segment_article(html)
    print(json.dumps(result, indent=2, ensure_ascii=False))
