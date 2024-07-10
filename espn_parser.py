from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import csv


def click_element(driver, element):
    try:
        element.click()
    except Exception as e:
        print(f"Click failed with error: {e}. Trying JavaScript click.")
        driver.execute_script("arguments[0].click();", element)
def get_play_component_data(page_url):
    # Initialize WebDriver (make sure to specify the path to your WebDriver)
    driver = webdriver.Chrome()

    # Open the page URL
    driver.get(page_url)

    # Wait for the content to load
    driver.implicitly_wait(10)  # Adjust the sleep time as needed

    quarter_texts = ['1st', '2nd', '3rd', '4th']
    driver.execute_script("window.scrollBy(0, 200);")

    # Loop through each quarter tab and extract data
    play_components_text = [["Time", "Play", "Quarter"]]
    for quarter_text in quarter_texts:
        # Click on the quarter button
        quarter_button = driver.find_element(By.XPATH,
                                             f'//button[contains(@class, "Button--unstyled tabs__link") and text()="{quarter_text}"]')
        quarter_button.click()
        driver.implicitly_wait(5)  # Adjust the sleep time as needed

        # Get the page source after the dynamic content is loaded
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        play_times_q = soup.find_all(True, class_=['playByPlay__time Table__TD'])
        play_info_q = soup.find_all(True,
                                  class_=['playByPlay__text tl Table__TD', 'playByPlay__text tl clr-btn Table__TD'])
        if not play_times_q:
            print("Parsing Error 1!" + quarter_text)
            return -1
        if not play_info_q:
            print("Parsing Error 2!" + quarter_text)
            return -1
        '''if len(play_times) != len(play_info):
            print("Parsing Error 3!")
            return -1
            '''

        play_times_text = [play_time.get_text(strip=True) for play_time in play_times_q]
        play_info_text = [info.get_text(strip=True) for info in play_info_q]
        for play_time, play_info in zip(play_times_text, play_info_text):
            play_components_text.append([play_time, play_info, quarter_text])

    if not play_components_text:
        print("No play components found. Verify the class name or structure of the HTML.")
    else:
        print(f"Found {len(play_components_text)} plays.")

    return play_components_text

def main():

    page_url = input("Enter ESPN game urls or END to finish the program\n")
    while page_url != "END":
        play_data = get_play_component_data(page_url)
        if play_data == -1:
            print("FATAL ERROR")
            return
        #save_to_file(play_data, 'play_components.txt')
        with open('output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(play_data)
        print(f"Play component text saved to output.csv")
        page_url = input()

if __name__ == '__main__':
    main()
