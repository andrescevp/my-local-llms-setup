import os
from time import sleep
from bs4 import Comment
from slugify import slugify
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
pwd = os.getcwd()
docs_folder = f"{pwd}/knowledge/php/limesurvey"

# create folder if not exists
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()

    # go to url
    page.goto("https://manual.limesurvey.org/LimeSurvey_Manual")
    # if element #CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll is visible then click
    if page.is_visible("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"):
        # click in #CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll button
        page.click("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")

    content = page.content()
    soup = BeautifulSoup(content)

    # get all <ul> elements between element #Manual_-_Table_of_Contents and #Translating_LimeSurvey
    from_element = soup.select("#Manual_-_Table_of_Contents")[0].parent
    to_element = soup.select("#Translating_LimeSurvey")[0].parent

    # get all <ul> elements between from_element and to_element
    ul_elements = []
    for element in from_element.next_siblings:
        if element == to_element:
            break
        if element.name == "ul":
            ul_elements.append(element)

    # get all <a> elements from ul_elements
    a_elements = []
    for ul_element in ul_elements:
        a_elements.extend(ul_element.find_all("a"))

    print(a_elements, ul_elements, from_element, to_element)

    # navigate to each link
    for a_element in a_elements:
        page.goto(f'https://manual.limesurvey.org{a_element["href"]}')
        if page.is_visible("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"):
            # click in #CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll button
            page.click("#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")
        # page_content = page.content()
        # bs_page = BeautifulSoup(page_content)
        # page_from_element = bs_page.select("#toc")[0]
        # page_to_element = bs_page.select("footer")[0]
        # # get all elements between page_from_element and page_to_element
        # content_elements = []
        # for element in page_from_element.next_siblings:
        #     # if comment, skip
        #     if isinstance(element, Comment):
        #         continue
        #     content_elements.append(element)
        #
        # # join all and get string
        # content_string = "".join([str(element) for element in content_elements])
        # # print(content_string)

        # click in #ca-viewsource
        # page.click("#actions-button")
        # get href visible #ca-viewsource > a and navigate to it
        view_source_btn = page.query_selector("#ca-viewsource")
        if not view_source_btn:
            continue
        code_view_html = view_source_btn.inner_html()
        code_view_soup = BeautifulSoup(code_view_html)
        code_view_href = code_view_soup.select("a")[0]["href"]
        page.goto(f'https://manual.limesurvey.org{code_view_href}')

        # page.click("#ca-viewsource")
        # get text in #wpTextbox1
        code_page_content = page.content()
        code_page_soup = BeautifulSoup(code_page_content)
        code_text = code_page_soup.select("#wpTextbox1")[0].text
        # save to knowledge/php/limesurvey
        with open(f"{docs_folder}/{slugify(a_element.text)}.txt", "w") as f:
            f.write(code_text)


    # get HTML
    # print(page.content())