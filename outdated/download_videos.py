import os
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
import time


def wait_for_downloads(download_dir, timeout=60):
    seconds = 0
    while seconds < timeout:
        if not any(fname.endswith('.crdownload') for fname in os.listdir(download_dir)):
            return True
        time.sleep(1)
        seconds += 1
    return False

def refresh_page(driver, page_url):
    driver.get(page_url)
    time.sleep(10)  # Adjust the sleep time as needed

def click_element(driver, element):
    try:
        element.click()
    except Exception as e:
        print(f"Click failed with error: {e}. Trying JavaScript click.")
        driver.execute_script("arguments[0].click();", element)

def handle_alert(driver):
    try:
        alert = WebDriverWait(driver, 10).until(EC.alert_is_present())
        print(f"Alert text: {alert.text}")
        alert.accept()
        time.sleep(2)  # Wait a bit after dismissing the alert
    except Exception as e:
        print(f"No alert present: {e}")

def download_highlights(page_url):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": "/Users/galishai/Downloads",  # Change this to your download directory
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    #chrome_options.add_argument('headless')

    # Initialize WebDriver
    service = Service('/Users/galishai/Desktop/AI Project/AI_Project/AI_PROJECT_SPORTS_HIGHLIGHTS/chromedriver')  # Change this to the path of your WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the page URL
        driver.get(page_url)

        # Wait for the page to load
        time.sleep(10)  # Adjust the sleep time as needed

        # Find all carousel items (assuming the carousel items have a specific class, adjust if necessary)
        carousel_items = driver.find_elements(By.CSS_SELECTOR,
                                              '.play-container')  # Replace with the actual class or selector

        print(f"Found {len(carousel_items)} carousel items.")
        total_items = len(carousel_items)
        # Click each carousel item and then the download button
        for index in range(0, total_items):
            try:
                # Scroll the carousel item into view
                driver.execute_script("arguments[0].scrollIntoView();", carousel_items[index])
                # Wait for the carousel item to be clickable
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(carousel_items[index]))
                click_element(driver, carousel_items[index])
                time.sleep(2)  # Adjust the sleep time as needed for the video to load

                # Find the download button within the current carousel item (adjust the selector as needed)
                download_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.download-video-button')))

                # Click the download button
                download_button.click()
                print(f"Clicked download button for item {index + 1}")

                # Wait for the download to complete
                if wait_for_downloads("/Users/galishai/Downloads"):
                    print(f"Download for item {index + 1} completed.")
                else:
                    print(f"Download {index + 1} did not complete in time.")
            except Exception as e:
                print(f"Error with carousel item {index + 1}: {e}")
                handle_alert(driver)  # Handle any alerts that appear
                print("Refreshing the page...")
                refresh_page(driver, page_url)
                time.sleep(10)  # Wait for the page to load
                carousel_items = driver.find_elements(By.CSS_SELECTOR,
                                                      '.play-container')  # Re-find all carousel items after refresh


    finally:
        # Close the WebDriver
        driver.quit()

def main():
    page_url = 'https://thehighlow.io/video/ids?ids=24zcKM24q5aY24bGEy24tNKv24znTy245wjc2422PS240eO924LaVr24Evjv24TxN424fZMd24YP3t24f2Fh24PQeO24jVPE24mN5P24S3Eg24KYEG24WCJs24REEp240HCk24Y7E124hovc24AxfI24wEZ024fjZw24y1jq24wlR824jdmx24A8LZ24IoXX24gR2224ypj124nqZz24FHA024D9aL24uGwb24HBxc24c2tk24aoj224Rw3Z24VPE324UhhD246IzE24C6bL24vatm24yf0S24CWsC249XFU24ae5q24kzyL24EJP924aHs824hbxC24NyoF24DsHD24BfI724SyH724R9Ig241Fae24JHVO24q7he24D2pk240tvU24yGu12490Yu244Q5024RScb24Y5jn24JG7L24Lqa424N8dV24KINL242J8N24lERM24ma6P24tl0X24qVSK24g9W724WVGx24b7qs24rPio24ffFF24uYOG24GLai244kOV24W1x024OIHW24SpN924ro0Y24JqgS24uNoV24uTDw240ERK24LGbC248uJg24BjK424dfx6246t9X24VJoG24NVFH24OQiu24LiKE24v5z924j8jU24QBKZ24WP4724jF3q24BHR3245UN424IfVi244yDU247jdA24i1NU24n0mV24mLX824MCkQ24G3IA24NUhi24oRws24J6fU24LriJ243RN724nIqT24BSMD24dWGV24H2Fc24mURE24egB624VTbG24UZdY24ScWl248Dy324HrFY24JMIq24bsWa24vqx0244N0N24kZj1240XPy24Zfsa24OEgy24875f24L3f824e2Tg24fMSA243uR124EyZP24oO6624bFtv24TIJW243q8M24f3oV24DI0k24aGiU24cmwN24wm9l24qcrk24Xzyb24PxRw24jo3X24BqQq24Pna224ilZu24iqXT24SCVh2415Vi24nPi224dxtU24UvDO24ANjl24iLmZ247X1e24qXjX243rZX24HA1m24I2eJ24LMk024lt4v243cKl24jidV246HQ424CNPB24yWuJ24nvLj249EeE24blw224bStE247Jpd24qwyA24nH1C24ypTr24cQAk24jowi24cWlf24jT1t24WllO2424OQ248A9B240r4c2481Cj24TBO024Se3r24MP3J240PYR24JM7524llct24PoIh24BZ2224cqBf24SDwX24mNmc24P2Ig24FZMB24uh7G24j5Md24QKoz24eE5G24dFiD24MtVh24j6NS24bSOw24hOQs24Mdfe24QmNE245LeC24RTS324qmTL24M0Io24jKkj24tCjJ24RkJA'
    download_highlights(page_url)

if __name__ == '__main__':
    main()
