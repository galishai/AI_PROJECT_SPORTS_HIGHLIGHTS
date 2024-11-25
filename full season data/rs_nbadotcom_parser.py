from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('capabilities={"acceptInsecureCerts": True}')

BUTTON_COUNT = 6
LINKS_TO_EXCLUDE = []







def main():
    page_url = "https://web.archive.org/web/20240415023755/https://www.nba.com/schedule?cal=all&pd=false&region=1&season=Regular%20Season"
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(page_url)
    WebDriverWait(driver, 10).until(
        lambda driver: driver.execute_script("return document.readyState") == "complete"
    )
    print("Page fully loaded!")
    count = 0
    target_data_text = "BOX SCORE"

    # Locate all elements with the specific data-text value

    while count < BUTTON_COUNT:
        #try:
        # Wait for the button to appear in the DOM
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        button = driver.find_element(By.XPATH, '//button[text()="LOAD MORE"]')

        # Scroll the button into view
        #driver.execute_script("arguments[0].scrollIntoView(true);", button)

        # Optionally highlight the button (for debugging purposes)
        #driver.execute_script("arguments[0].style.border='3px solid red'", button)

        # Click the button
        button.click()
        count += 1
        print("Button "+ str(count) +" clicked successfully!")
        #except Exception as e:
            #print(f"Error: {e}")
    elements = driver.find_elements(By.CSS_SELECTOR, f'[data-text="{target_data_text}"]')
    box_score_links = []
    for element in elements:
        href = element.get_attribute("href")
        if href and href not in LINKS_TO_EXCLUDE:
            box_score_links.append(href)
    print("links found: " + str(len(box_score_links)))
    with open('rs_nbadotcom_links.txt', mode='w', newline='') as file:
        for link in box_score_links:
            file.write(link + "\n")  # Add a newline after each string
    print(f"links saved")
    print("OK :)")
    driver.quit()




if __name__ == '__main__':
    main()

