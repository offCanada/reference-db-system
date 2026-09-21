from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainMatch:
    domain: str
    matched_rule: str | None
    confidence: float
    status: str


def _is_negated(title_lower: str, match_start: int, match_end: int) -> bool:
    prefix = title_lower[max(0, match_start - 25):match_start]

    # Words that are food items — negation of these means the product is NOT food.
    # Note: "gluten" is NOT here because "gluten-free bread" is still bread.
    food_words = {
        "chicken", "beef", "pork", "turkey", "lamb", "veal", "fish",
        "salmon", "tuna", "shrimp", "seafood", "meat",
        "milk", "cheese", "yogurt", "butter", "cream", "egg",
        "bread", "cake", "cookie", "pastry", "pie",
        "juice", "soda", "coffee", "tea",
        "pasta", "rice", "noodle",
        "fruit", "vegetable", "potato", "tomato", "apple", "banana",
    }

    # "no" / "without" directly before the match = negation
    # (the food word IS the match, so prefix is just "no " / "without ")
    if re.search(r"\b(?:no|without)\s+$", prefix):
        return True

    # "non-X" prefix is always negation
    if re.search(r"\bnon[\s-]+$", prefix):
        return True

    # "X free" pattern: only negate if X is a food word
    # Handles both "gluten free bread" and "free gluten bread" in prefix
    free_match = re.search(r"\b(\w+)[\s-]*free\b", prefix)
    if free_match:
        word_before_free = free_match.group(1)
        return word_before_free in food_words

    # "zero X" pattern: only negate if X is a food word
    zero_match = re.search(r"\bzero\s+(\w+)\s*$", prefix)
    if zero_match:
        word_after_zero = zero_match.group(1)
        return word_after_zero in food_words

    return False


DISAMBIGUATION_RULES = [
    (
        r"\bpaper\s+bag\b.*\b(?:potato|potatoes|apple|bananas?|orange|carrot|onion|garlic)\b",
        "food",
        0.9,
    ),
    (
        r"\b(?:potato|potatoes|apple|bananas?|orange|carrot|onion|garlic)\b.*\bpaper\s+bag\b",
        "food",
        0.9,
    ),
    (
        r"\bpaper\s+bag\s+(?:potato|potatoes)\b",
        "food",
        0.9,
    ),
    (
        r"\bpaper\s+bag\s+(?:red|yellow|white|brown)\b",
        "food",
        0.9,
    ),
    (
        r"\bice\s+cream\s+cups?\b",
        "ambiguous",
        0.5,
    ),
    (
        r"\b(?:bird|wild\s+bird|feeder)\s+(?:feed|food|seed|sunflower)\b",
        "non_food",
        0.9,
    ),
    (
        r"\b(?:wild|bird)\b.*\b(?:seed|feed)\b",
        "non_food",
        0.8,
    ),
    (
        r"\bnyjer\s+seed\b",
        "non_food",
        0.9,
    ),
    (
        r"\b(?:cat|dog|pet)\b.*\b(?:food|treat|dental|stick|chew|kibble)\b",
        "non_food",
        0.9,
    ),
    (
        r"\b(?:cat|dog|pet)\b.*\b(?:chicken|beef|turkey|salmon)\b",
        "non_food",
        0.8,
    ),
    (
        r"\bcastor\s+oil\b",
        "non_food",
        0.9,
    ),
    (
        r"\bfurniture\s+polish\b",
        "non_food",
        0.9,
    ),
    (
        r"\b(?:multi\s+surface|surface)\s+disinfectant\b",
        "non_food",
        0.9,
    ),
    (
        r"\bhot\s+water\s+bottle\b",
        "non_food",
        0.9,
    ),
    (
        r"\bwater\s+bottle\b",
        "non_food",
        0.8,
    ),
    (
        r"\bcoffee\s+filters?\b",
        "non_food",
        0.9,
    ),
    (
        r"\bcone\s+coffee\s+filters?\b",
        "non_food",
        0.9,
    ),
    (
        r"\bfilters?\s+coffee\b",
        "non_food",
        0.9,
    ),
    (
        r"\bchristmas\s+crackers?\b",
        "non_food",
        0.9,
    ),
    (
        r"\bcrackers?\s+\d+\s*-?\s*inch\b",
        "non_food",
        0.9,
    ),
    (
        r"\bwhitening\s+wraps?\b",
        "non_food",
        0.9,
    ),
    (
        r"\bextreme\s+whitening\b",
        "non_food",
        0.9,
    ),
    (
        r"\bflour\s+sack\s+towels?\b",
        "non_food",
        0.9,
    ),
    (
        r"\bdenture\s+cleanser\b",
        "non_food",
        0.9,
    ),
    (
        r"\bcorn\s+gel\s+pads?\b",
        "non_food",
        0.9,
    ),
    (
        r"\bpads?\s+corn\s+gel\b",
        "non_food",
        0.9,
    ),
    (
        r"\bgel\s+pads?\b",
        "non_food",
        0.8,
    ),
    (
        r"\bzaz\b.*\b(?:water|enhancer)\b",
        "non_food",
        0.9,
    ),
    (
        r"\bwater\s+enhancer\b",
        "non_food",
        0.8,
    ),
    (
        r"\bcedar\s+grilling\s+plank\b",
        "non_food",
        0.9,
    ),
    (
        r"\bgrilling\s+plank\b",
        "non_food",
        0.8,
    ),
    (
        r"\bsnack\s+bags?\b",
        "non_food",
        0.9,
    ),
    (
        r"\bgingerbread\s+(?:house|building)\s+kit\b",
        "non_food",
        0.9,
    ),
    (
        r"\bpre-built\s+gingerbread\b",
        "non_food",
        0.9,
    ),
    (
        r"(?=.*\bgingerbread\b)(?=.*\b(?:house|building|train)\b)(?=.*\bkit\b)",
        "non_food",
        0.9,
    ),
    (
        r"\bcups?\s+\d+\s*pack\b",
        "non_food",
        0.7,
    ),
    (
        r"\bbird\s+feed\b",
        "non_food",
        0.9,
    ),
    (
        r"\b(?:cat|dog|pet)\s+food\b",
        "non_food",
        0.9,
    ),
    (
        r"\b(?:vitamin|supplement|probiotic)\b",
        "non_food",
        0.9,
    ),
    (
        r"\b(?:capsules?|tablets?|caplets?|softgels?|gelcaps?)\b",
        "non_food",
        0.8,
    ),
]


NONFOOD_SPECIFIC = [
    (r"\b(?:cat|dog|pet)\s+(?:\w+\s+)?(?:food|treat)\b", "non_food"),
    (r"\b(?:puppy|kitten)\s+food\b", "non_food"),
    (r"\b(?:ibuprofen|acetaminophen|melatonin)\b", "non_food"),
    (r"\b(?:hydrogen\s+peroxide|hydrocortisone|clotrimazole|diphenhydramine)\b", "non_food"),
    (r"\b(?:antibiotic|antifungal)\s+(?:ointment|cream)\b", "non_food"),
    (r"\b(?:calamine|lotion|anti-itch)\b", "non_food"),
    (r"\b(?:prenatal|pregnancy\s+test)\b", "non_food"),
    (r"\b(?:vitamin|supplement|probiotic)\s+(?:capsules?|tablets?|softgels?|gummies?)\b", "non_food"),
    (r"\b\d+\s*(?:mg|iu)\b.*\b(?:capsules?|tablets?|gelcaps?|softgels?)\b", "non_food"),
    (r"\b(?:acid\s+reducer|eye\s+care|lutein|omega\s+softgels?)\b", "non_food"),
    (r"\b(?:muscle|joint)\s+relief\b", "non_food"),
    (r"\b(?:tampons?|maxi\s+pads?|sanitary\s+pads?|bladder\s+protection)\b", "non_food"),
    (r"\b(?:toothbrushes?|toothpastes?|shampoos?|conditioners?|deodorants?)\b", "non_food"),
    (r"\b(?:sunscreens?|moisturizing\s+(?:lotion|shampoo|conditioner))\b", "non_food"),
    (r"\b(?:protective\s+underwear|underwear|toe\s+nail\s+clippers?|ear\s+plugs?)\b", "non_food"),
    (r"\b(?:nail\s+polish\s+remover|acetone|cotton\s+pads?)\b", "non_food"),
    (r"\b(?:detergents?|laundry|fabric\s+softeners?|bleach)\b", "non_food"),
    (r"\b(?:garbage|recycling)\s+bags?\b", "non_food"),
    (r"\b(?:paper\s+towels?|facial\s+tissues?|bathroom\s+tissues?|napkins?)\b", "non_food"),
    (r"\b(?:dish\s+(?:detergent|cloth|drying)|dishwashers?)\b", "non_food"),
    (r"\b(?:sponges?|scouring\s+pads?|dusters?|mopping|mops?|scrubbers?)\b", "non_food"),
    (r"\b(?:cloth\s+(?:reusable|cleaning)|fire\s+logs?|epsom\s+salts?)\b", "non_food"),
    (r"\b(?:compostable\s+(?:bin\s+)?liners?|plastic\s+straws?|liners?\s+(?:liner|bag)|bin\s+liners?)\b", "non_food"),
    (r"\b(?:shav(?:e|ing)|hand|body|face|moisturiz(?:er|ing)|diaper|antibiotic)\s+creams?\b", "non_food"),
    (r"\b(?:light\s+bulbs?|bulbs?|foil\s+containers?)\b", "non_food"),
    (r"\b(?:muffin|pizza|cake|baking)\s+pans?\b", "non_food"),
    (r"\b(?:skewers?|lunch\s+bags?|pill\s+(?:reminder|planner|box))\b", "non_food"),
    (r"\b(?:medication\s+organizers?|syringes?|eye\s+and\s+ear)\b", "non_food"),
    (r"\b(?:latex\s+(?:gloves?|finger\s+cots?)|finger\s+cots?)\b", "non_food"),
    (r"\b(?:baby\s+wipes?|cotton\s+balls?|floss|mouthwash|razor|shave\s+gel)\b", "non_food"),
    (r"\b(?:body\s+wash|hand\s+soap|petroleum|salon\s+boards?|diapers?)\b", "non_food"),
    (r"\b(?:toilet\s+paper|paper\s+plates?|foam\s+cups?|disposable)\b", "non_food"),
    (r"\b(?:plastic\s+(?:bags?|wrap|straws?|cutlery)|wax\s+paper|aluminum\s+foil)\b", "non_food"),
    (r"\b(?:cleaning\s+eraser|scrubbers?|containers?|wrap|litter|compost|mulch)\b", "non_food"),
    (r"\b(?:bird\s+food|cat\s+litter|men's\s+razor|women's\s+razor|footcare)\b", "non_food"),
    (r"\b(?:bath\s+soak|miconazole|anti-nausea|wristband|large\s+blade)\b", "non_food"),
    (r"\b(?:non\s+stick|loaf\s+pan|foam\s+bowl|licecomb|lotion|razors?)\b", "non_food"),
    (r"\b(?:cartridges?|blades|nitrile\s+gloves?|silicone\s+(?:mitten|glove))\b", "non_food"),
    (r"\b(?:oven\s+mitten|peelers?|graters?|can\s+openers?|scissors?|funnels?)\b", "non_food"),
    (r"\b(?:timers?|thermometers?|scales?|mixing\s+bowl|measuring\s+(?:cups?|spoons?))\b", "non_food"),
    (r"\b(?:spatulas?|ladles?|tongs|whisks?|rolling\s+pin|cookie\s+cutters?)\b", "non_food"),
    (r"\b(?:colanders?|strainers?|trivets?|potholders?|coasters?|placemats?)\b", "non_food"),
    (r"\b(?:tablecloths?|napkin\s+rings?|watering\s+cans?|garden\s+tools?|planters?)\b", "non_food"),
    (r"\b(?:pots\s+plants?|fertilizer|weed\s+killer|insect\s+repellent)\b", "non_food"),
    (r"\b(?:flashlights?|batteries?|candles?|matches?|fire\s+extinguishers?)\b", "non_food"),
    (r"\b(?:smoke\s+detectors?|air\s+freshener|fabric\s+refresher|stain\s+remover)\b", "non_food"),
    (r"\b(?:odor\s+eliminator|cleaner|bandages?|shave|mouthwash)\b", "non_food"),
    (r"\b(?:foam\s+(?:cups?|bowls?)|plastic\s+(?:knife|fork|spoon|cutlery))\b", "non_food"),
    (r"\b(?:diapers?|razor|cartridge|toothbrush|nail\s+brush|washing\s+machine)\b", "non_food"),
    (r"\b(?:floor\s+cleaner|anti-bacterial|antiseptic|bath\s+soak|lice)\b", "non_food"),
    (r"\b(?:nitrile|silicone\s+(?:mitten|glove)|oven\s+mitten|sandwich\s+bags?)\b", "non_food"),
    (r"\b(?:kitchen\s+bags?|compost|mulch|containers?|tote)\b", "non_food"),
    (r"\b(?:isopropyl|bandage|allergy)\b", "non_food"),
    (r"\b(?:softgels?|footcare|bath\s+soak|miconazole|anti\s+nausea|wristband)\b", "non_food"),
    (r"\b(?:cold\s+(?:and\s+)?(?:sinus|flu|medicine|relief)|sinus\s+(?:medication|relief)|head\s+cold)\b", "non_food"),
    (r"\b(?:pain\s+relief|muscle\s+(?:relief|aches|and\s+back)|joint\s+relief|back\s+pain)\b", "non_food"),
    (r"\b(?:caplets?|tablets?)\b", "non_food"),
    (r"\b(?:diarrhea\s+relief|antacid|digestive\s+(?:comfort|extra))\b", "non_food"),
    (r"\b(?:cough|cold)\s+(?:relief|medicine)\b", "non_food"),
    (r"\b(?:dish(?:washing)?\s+(?:liquid|detergent)|dish\s+liquid)\b", "non_food"),
    (r"\b(?:plastic\s+(?:forks?|spoons?|knives?|cutlery))\b", "non_food"),
    (r"\b(?:foam\s+plates?|paper\s+(?:plates?|bowls?))\b", "non_food"),
    (r"\b(?:tweezers?|toenail\s+clippers?|fingernail\s+clippers?|nail\s+clippers?)\b", "non_food"),
    (r"\b(?:shaving\s+brush|sleep\s+mask|after\s+sun|sanitizing\s+gel)\b", "non_food"),
    (r"\b(?:replacement\s+brush|brush\s+heads?|finger\s+splint|foot\s+smoother|spacers?)\b", "non_food"),
    (r"\b(?:lint\s+roller|oven\s+mitts?|oven\s+mitt)\b", "non_food"),
    (r"\b(?:training\s+pants?|non-latex\s+vinyl\s+gloves?)\b", "non_food"),
    (r"\b(?:sweeping\s+cloths?|parchment\s+paper|first\s+aid\s+tape)\b", "non_food"),
    (r"\b(?:flex\s+straws?|paper\s+straws?|bamboo\s+picks?)\b", "non_food"),
    (r"\b(?:scours?|buffer(?!.*(?:chicken|beef|pork))|file(?!.*(?:chicken|beef|pork))|bracelet)\b", "non_food"),
    (r"\b(?:70\s+litres|mega\s+soil|topsoil)\b", "non_food"),
    (r"\bcheese\s+cloth\b", "non_food"),
    (r"\b(?:cough\s+lozenge|lozenges?)\b", "non_food"),
    (r"\bfirst\s+aid\s+kit\b", "non_food"),
    (r"\breusable\s+bag\b", "non_food"),
    (r"\bface\s+mask\b", "non_food"),
    (r"\bclay\s+mask\b", "non_food"),
    (r"\bsheet\s+mask\b", "non_food"),
    (r"\bbaking\s+sheet\b", "non_food"),
    (r"\blasagna\s+pan\b", "non_food"),
    (r"\bspringform\b", "non_food"),
    (r"\b(?:baking\s+(?:sheet|pan)|lasagna\s+pan|springform\s+(?:cake\s+)?pan)\b", "non_food"),
    (r"\bgarden\s+soil\b", "non_food"),
    (r"\bcooking\s+sheet\b", "non_food"),
    (r"\bultra\s+thin\s+pads?\b", "non_food"),
    (r"\bpads?\s+(?:with\s+wings?|long|super|overnight)\b", "non_food"),
    (r"\b(?:wound\s+)?dressing\s+strips?\b", "non_food"),
    (
        r"\b(?:tea|paper|plastic|foam|kitchen)\s+(?:towels?|filters?|plates?|cups?|bowls?|cutlery|wrap)\b",
        "non_food",
    ),
    (r"\b(?:coffee|paper|plastic|foam|kitchen)\s+bags?\b", "non_food"),
    (r"\b(?:tea)\s+towels?\b", "non_food"),
    (r"\b(?:dish|laundry|floor|glass|multi-surface)\s+(?:detergent|soap|cleaner|washing\s+liquid)\b", "non_food"),
    (r"\b(?:air|fabric|odor)\s+(?:freshener|eliminator|refresher)\b", "non_food"),
    (r"\b(?:garbage|compost|recycling)\s+(?:bags?|liners?|cans?)\b", "non_food"),
    (r"\b(?:aluminum|plastic|wax|parchment)\s+(?:foil|wrap|paper)\b", "non_food"),
    (r"\b(?:hand|body|face|foot|eye)\s+(?:soap|lotion|cream|wash|mask)\b", "non_food"),
    (r"\b(?:vitamin|mineral|supplement|probiotic|protein)\s+(?:capsules?|tablets?|softgels?|gummies?)\b", "non_food"),
    (r"\b(?:pain|headache|sinus|allergy|cold|flu|sleep)\s+(?:relief|medicine|medication|tablets?|capsules?)\b", "non_food"),
    (r"\b(?:baby|infant|toddler)\s+(?:wipes?|diapers?|shampoo|lotion|cream|oil)\b", "non_food"),
    (r"\b(?:menstrual|sanitary|incontinence)\s+(?:pads?|underwear|liners?)\b", "non_food"),
    (r"\btea\s+towel\b", "non_food"),
    (r"\bcoffee\s+filter\b", "non_food"),
    (r"\bdisinfecting\s+wipes?\b", "non_food"),
    (r"\b(?:fibre|fiber)\s+lax\b", "non_food"),
    (r"\bnicotine\b", "non_food"),
    (r"\bwine\s+glasses?\b", "non_food"),
    (r"\bpaper\s+bag\b", "non_food"),
    (r"\bcompostable\s+(?:birch\s+wood\s+)?(?:mix\s+)?cutlery\b", "non_food"),
    (r"\beye\s+glass\s+repair\b", "non_food"),
    (r"\bplastic\s+(?:beer|wine)\s+(?:cup|glass)\b", "non_food"),
    (r"\bnighttime\s+(?:cold|honey)\b", "non_food"),
    (r"\banti\s+smoking\b", "non_food"),
    (r"\bnicotine\s+gum\b", "non_food"),
    (r"\b(?:mineral|essential|baby)\s+oil\b", "non_food"),
    (r"\bbismuth\b", "non_food"),
    (r"\bsupplements?\b.*\bgummies?\b", "non_food"),
    (r"\bgummies?\b.*\bsupplements?\b", "non_food"),
]


FOOD_SPECIFIC = [
    (r"\btea\s+bags?\b", "food"),
    (r"\b(?:yogurts?|cottage\s+cheese|cream\s+cheese|sour\s+cream|milk|butter|margarine)\b", "food"),
    (r"\b(?:cheese|mozzarella|cheddar|parmesan|ricotta|cream|whipping\s+cream|half\s+and\s+half)\b", "food"),
    (r"\b(?:chickens?|beefs?|porks?|sausages?|wieners?|bacons?|turkeys?|lambs?)\b", "food"),
    (r"\b(?:salmons?|tunas?|shrimps?|fish|seafood)\b", "food"),
    (r"\b(?:breads?|bagels?|muffins?|croissants?|tortillas?|wraps?|flatbreads?|pitas?|naans?)\b", "food"),
    (r"\b(?:cinnamon\s+rolls?|donuts?|pies?|cookies?|cakes?|brownies?|pastries?)\b", "food"),
    (r"\b(?:buns?|biscuits?|crackers?)\b", "food"),
    (r"\b(?:juices?|water|coffees?|teas?|sodas?|pops?|sport\s+drinks?|energy\s+drinks?)\b", "food"),
    (r"\b(?:drinks?|cream\s+soda|lemonades?|cola)\b", "food"),
    (r"\b(?:pastas?|noodles?|rices?|quinoas?|oats?|cereals?|granolas?|flours?|sugars?|salts?)\b", "food"),
    (r"\b(?:spices?|seasonings?|vinegars?|oils?|sauces?|ketchups?|mustards?)\b", "food"),
    (r"\b(?:mayonnaises?|peanut\s+butter|jams?|honeys?|syrups?|baking|yeast|cocoa)\b", "food"),
    (r"\b(?:chocolates?|candies?|gumm[ie]s?|lollipops?|liquorices?|licorices?|raisins?)\b", "food"),
    (r"\b(?:coleslaws?|frozen|pizzas?|ice\s+cream|sorbets?|tacos?|taco\s+shells?)\b", "food"),
    (r"\b(?:apples?|bananas?|oranges?|grapes?|strawberr(?:y|ies)|blueberr(?:y|ies)|cranberr(?:y|ies))\b", "food"),
    (r"\b(?:lemons?|limes?|peaches?|mangos?|pineapples?|cherries?|avocados?)\b", "food"),
    (r"\b(?:tomato(?:es)?|onions?|garlics?|carrots?|celery|lettuces?|romaines?|salads?|greens\b)\b", "food"),
    (r"\b(?:vegetables?|brussels?|broccolis?|spinachs?|kales?|potato(?:es)?|sweet\s+potatoes?)\b", "food"),
    (r"\b(?:chips?|popcorns?|nuts?|almonds?|cashews?|walnuts?|peanuts?|pistachios?|pecans?)\b", "food"),
    (r"\b(?:sunflower\s+seeds?|trail\s+mix|snack\s+mix|granola\s+bars?|protein\s+bars?)\b", "food"),
    (r"\b(?:nut\s+bars?|fruit\s+snacks?)\b", "food"),
    (r"\b(?:dips?|salsas?|marinades?|dressings?|spreads?|relishes?|pickles?|olives?|capers?)\b", "food"),
    (r"\b(?:mushrooms?|thyme|guacamole|hummus|croutons?|ham|salami|pepperoni|prosciutto)\b", "food"),
    (r"\b(?:bacon|sausage|deli|luncheon|egg|couscous|gnocchi|barley|prunes?|marshmallow)\b", "food"),
    (r"\b(?:candy|jelly\s+gums?|gummies|jujubes?|wine\s+gums?|juice|beer|ale|ginger\s+ale)\b", "food"),
    (r"\b(?:cocktail|cranberry|non-alcoholic|non\s+alcoholic|mint|arugula|spring\s+mix)\b", "food"),
    (r"\b(?:salad|coconut|vanilla\s+extract|cinnamon|peppercorn|corn|beans?|peas?)\b", "food"),
    (r"\b(?:peppers?|cucumber|lettuce|spinach|broccoli|cabbage|cauliflower|celery|asparagus)\b", "food"),
    (r"\b(?:kale|yams?|rutabaga|parsnip|turnip|radish|beets?|pumpkin|squash|zucchini)\b", "food"),
    (r"\b(?:horseradish|ginger|pears?|plums?|kiwi|mango|pineapple|papaya|pomegranate|fig)\b", "food"),
    (r"\b(?:date|cranberr(?:y|ies)|clementines?|mandarins?|canned|soup|lentil|split\s+pea|chili)\b", "food"),
    (r"\b(?:stew|broth|stock|paste|sauce|vinegar|oil|sugar|honey|syrup|flour|rice)\b", "food"),
    (r"\b(?:oat|cereal|granola|pancake|waffle|pie|tart|cheesecake|pastry|croissant)\b", "food"),
    (r"\b(?:cookie|cake|brownie|muffin|bread|bagel|pretzel|crackers?|rusks?)\b", "food"),
    (r"\b(?:popcorn|chips?|nuts?|almonds?|cashews?|walnuts?|peanuts?|pistachios?|pecans?|macadamia|hazelnuts?)\b", "food"),
    (r"\b(?:sunflower\s+seeds?|trail\s+mix|party\s+mix|pub\s+mix|cocktail\s+mix|granola\s+bar|protein\s+bar)\b", "food"),
    (r"\b(?:chocolate|caramels?|fudge|lollipop|gummy|jube|jujube|taffy|licorice|confection|sweets?|baking)\b", "food"),
    (r"\b(?:coating|mix|kit|extract|lard|shortening|stevia|mayonnaise|mustard|ketchup|hot\s+sauce)\b", "food"),
    (r"\b(?:soy\s+sauce|teriyaki|sriracha|pickled|sauerkraut|kimchi|fermented|juice|cocktail)\b", "food"),
    (r"\b(?:cranberry|non\s+alcoholic|ginger\s+ale|beer|lager|ale|stout|porter|brandy|rum|whiskey|vodka|gin|tequila|wine|champagne)\b", "food"),
    (r"\b(?:coffee|tea|water|soda|drinks?|gum|jelly|toffee|marshmallow|mixed\s+nuts|cracker|pretzel|popcorn|chips|pasta|spaghetti|noodles|gnocchi|ravioli|lasagna)\b", "food"),
    (r"\b(?:pie|tart|cheesecake|cobbler|crisp|crumble|shortcake|cake|brownie|muffin|bread|croissant|cookie|bacon|sausage|ham|salami|pepperoni|chicken|beef|pork|turkey|lamb|fish|salmon|tuna|shrimp|lobster|crab|clam|mussel|oyster|egg|milk|cheese|yogurt|cream|butter|margarine|sour\s+cream|cottage\s+cheese|cream\s+cheese)\b", "food"),
    (r"\b(?:eggs?|marshmallows?|oatmeal|mussels?|chilies?|potato|sweetener|cauliettes?|sesame\s+seeds?|bologna|pudding|grains?|seed)\b", "food"),
    (r"\b(?:cooking\s+spray|sage|bay\s+leaves?|currants?|dessert|brittle|cucumbers?|mints?|nutmeg|flax\s+seeds?|melba\s+toast|toast|molasses|veal|gums?|snacks?|fruit|flavoured)\b", "food"),
    (r"\b(?:dates?|chia\s+seeds?|pretzels?|rosemary|dill|cilantro|jumbleberry|blend|meat|smoked|sliced|grilling|plank)\b", "food"),
    (r"\b(?:graham|crumbs?|broccolini|oregano|tarragon|chives?|macaroons?|chocolatey)\b", "food"),
    (r"\b(?:basil|peach\s+rings?|baguette|pastrami|grapefruit|frosting|relish|quiche|dried|apricots?|lentils?|savory|submarine|variety|italian|multigrain)\b", "food"),
    (r"\bcoffee\s+(?:cups?|pods?|k[\s-]?cups?)\b", "food"),
]


GENERIC_RULES = [
    (r"\b(?:snack|natural|simple|organic|dessert|food)\b", "food"),
    (r"\b(?:trail|snack|party|pub|cocktail|baking)\s+mix\b", "food"),
    (r"\b(?:baking|gift|sampler|variety|first\s+aid)\s+kit\b", "food"),
    (r"\bkit\b", "non_food"),
    (r"\b(?:lawn|leaf|sanitizer|protectors?|lens|nail|emery|nipper|ibu|eye\s+drops?|topsoil|swabs?|daytime)\b", "non_food"),
    (r"\b(?:cold\s+flu|flu|tablets?|patch(?:es)?|antidiarrheal|asa|lax|pill|splitter|crusher|liners?|fluid|washer|windshield|conazole|sachet|finger\s+covers?|reusable|disposable|compostable|recycle|recycling|waste|match(?:es)?|gauze|sterile|fluticasone|nasal|stool|softener|lavender|eucalyptus|mcg|dose)\b", "non_food"),
    (r"\b(?:bags?|cups?|glasses?|towels?|wipes?|capsules?|rub|spray|can|freezer|cutlery)\b", "non_food"),
]


def _count_required_words(pattern: str) -> int:
    clean = re.sub(r"\\b", "", pattern)
    clean = re.sub(r"\\s\+?", " ", clean)
    clean = re.sub(r"\(\?:.*?\)", "WORD", clean)
    clean = re.sub(r"\[.*?\]", "X", clean)
    clean = re.sub(r"[+*?]$", "", clean)

    words = [
        word
        for word in clean.split()
        if word and word not in ("|", ".*", ".+")
    ]

    return len(words)


def classify_product_domain(title: str) -> DomainMatch:
    text = str(title or "").lower()

    for pattern, domain, confidence in DISAMBIGUATION_RULES:
        match = re.search(pattern, text)

        if not match:
            continue

        if _is_negated(text, match.start(), match.end()):
            continue

        if domain == "ambiguous":
            return DomainMatch(
                domain="unknown",
                matched_rule=pattern,
                confidence=confidence,
                status="AMBIGUOUS",
            )

        return DomainMatch(
            domain=domain,
            matched_rule=pattern,
            confidence=confidence,
            status="PASS",
        )

    for pattern, domain in NONFOOD_SPECIFIC:
        match = re.search(pattern, text)

        if not match:
            continue

        if _is_negated(text, match.start(), match.end()):
            continue

        specificity = _count_required_words(pattern)

        if specificity >= 3:
            confidence = 0.9
        elif specificity == 2:
            confidence = 0.8
        else:
            confidence = 0.6

        return DomainMatch(
            domain=domain,
            matched_rule=pattern,
            confidence=confidence,
            status="PASS",
        )

    for pattern, domain in FOOD_SPECIFIC:
        match = re.search(pattern, text)

        if not match:
            continue

        if _is_negated(text, match.start(), match.end()):
            continue

        specificity = _count_required_words(pattern)

        if specificity >= 3:
            confidence = 0.9
        elif specificity == 2:
            confidence = 0.8
        else:
            confidence = 0.6

        return DomainMatch(
            domain=domain,
            matched_rule=pattern,
            confidence=confidence,
            status="PASS",
        )

    for pattern, domain in GENERIC_RULES:
        match = re.search(pattern, text)

        if not match:
            continue

        if _is_negated(text, match.start(), match.end()):
            continue

        return DomainMatch(
            domain=domain,
            matched_rule=pattern,
            confidence=0.5,
            status="WARNING",
        )

    return DomainMatch(
        domain="unknown",
        matched_rule=None,
        confidence=0.0,
        status="AMBIGUOUS",
    )