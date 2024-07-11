import scrapy

class MagicEdenRunesSpider(scrapy.Spider):
    name = 'magicedenrunes'
    start_urls = ['https://magiceden.io/runes']

    def parse(self, response):
        # Adjust the selectors based on the actual HTML structure of the page
        for rune in response.css('.card'):
            yield {
                'name': rune.css('.card-title::text').get(),
                'price': rune.css('.card-price::text').get(),
                'owner': rune.css('.card-owner::text').get(),
                'link': rune.css('a::attr(href)').get(),
                # Add more fields as needed
            }

        # Follow pagination links
        next_page = response.css('a.next-page::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

# To run the spider, save the above script as magiceden_runes_spider.py
# Then, use the following command in the terminal:
# scrapy runspider magiceden_runes_spider.py -o runes_data.json
