const pptr = require('puppeteer')

async function scrapeCoin(url){
    const browser = await pptr.launch()
    const page = await browser.newPage()
    await page.goto(url)

    const [el] = await page.$x('/html/body/div[1]/div/div[2]/div[1]/div[3]')
    const src = el.getProprety('src')

    const txt = await src.jsonValue()

    console.log(txt)
}

scrapeCoin('https://www.acsearch.info/search.html?term=+John+Hyrcanus+I&category=1-2&lot=&thesaurus=1&images=1&en=1&de=1&fr=1&it=1&es=1&ot=1&currency=usd&order=0')