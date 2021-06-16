import os
import selenium
from selenium import webdriver
import time
from PIL import Image
import io
import requests
from selenium.common.exceptions import ElementClickInterceptedException
# from webdriver_manager.chrome import ChromeDriverManager

if __name__ == '__main__':
    driver = webdriver.Chrome("/Users/anchitshrivastava/Downloads/chromedriver")
    # driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
    # Specify Search URL
    search_url = "https://www.google.com/search?q=advertisements&rlz=1C5CHFA_enIN816IN817&sxsrf=ALeKk034QV84cb4sXeFJtdgBZmRdVH-pCw:1623133981586&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiJn5SjtYfxAhVwzzgGHaNbB9QQ_AUoAXoECAEQAw&biw=1280&bih=647"
    driver.get(search_url)
    # Scroll to the end of the page
    for i in range(4):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
    # Locate the images to be scraped from the current page
    imgResults = driver.find_elements_by_xpath("//img[contains(@class,'Q4LuWd')]")
    totalResults = len(imgResults)
    # Click on each Image to extract its corresponding link to download

    img_urls = set()
    for i in range(0, len(imgResults)):
        img = imgResults[i]
        try:
            img.click()
            time.sleep(2)
            actual_images = driver.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'https' in actual_image.get_attribute('src'):
                    img_urls.add(actual_image.get_attribute('src'))
        except ElementClickInterceptedException or ElementNotInteractableException as err:
            print(err)

    os.chdir('/Users/anchitshrivastava/Desktop/Study/data science:data analytics chitkara/IP project Ads classification/Ads')
    baseDir = os.getcwd()

    for i, url in enumerate(img_urls):
        file_name = f"{i}.jpg"
        try:
            image_content = requests.get(url).content

        except Exception as e:
            print(f"ERROR - COULD NOT DOWNLOAD {url} - {e}")

        try:
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')

            file_path = os.path.join(baseDir, file_name)

            with open(file_path, 'wb') as f:
                image.save(f, "JPEG", quality=85)
            print(f"SAVED - {url} - AT: {file_path}")
        except Exception as e:
            print(f"ERROR - COULD NOT SAVE {url} - {e}")

