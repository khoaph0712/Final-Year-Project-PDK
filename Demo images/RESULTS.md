# Demo images

Real-world (non-dataset) trash photos, one folder per run through the deployed
WasteWise pipeline (`web/server.py`, YOLO26m + ConvNeXt-Tiny + 637 features).
Every image below passes: the headline material matches the true material.
Annotated copies with detection boxes are in `annotated/`.

All photos are free-license from Wikimedia Commons; sources and licenses:

| file | true class | pipeline result | bin | source | license |
|---|---|---|---|---|---|
| cardboard_2.jpg | cardboard | Cardboard 93% | Recycling | [File:Cardboard in Trashbin.jpg](https://commons.wikimedia.org/wiki/File:Cardboard_in_Trashbin.jpg) | CC BY-SA 4.0 |
| glass_2.jpg | glass | Glass 54% | Recycling | [File:Empty beer bottle on grass.jpg](https://commons.wikimedia.org/wiki/File:Empty_beer_bottle_on_grass.jpg) | CC BY-SA 4.0 |
| metal_4.jpg | metal | Metal 79% | Recycling | [File:Miller High Life flattened.jpg](https://commons.wikimedia.org/wiki/File:Miller_High_Life_flattened.jpg) | CC0 |
| organic_1.jpg | organic | Organic 35% | Compost | [File:Banana peel on the ground.jpg](https://commons.wikimedia.org/wiki/File:Banana_peel_on_the_ground.jpg) | CC BY-SA 4.0 |
| paper_7.jpg | paper | Paper 44% | Recycling | [File:Discarded newspaper, Cranny - geograph.org.uk - 985790.jpg](https://commons.wikimedia.org/wiki/File:Discarded_newspaper,_Cranny_-_geograph.org.uk_-_985790.jpg) | CC BY-SA 2.0 |
| plastic_2.jpg | plastic | Plastic 51% | Recycling | [File:A discarded plastic bottle and a surfer.jpg](https://commons.wikimedia.org/wiki/File:A_discarded_plastic_bottle_and_a_surfer.jpg) | CC BY-SA 4.0 |
| plastic_3.jpg | plastic | Plastic 43% | Recycling | [File:Coca-Cola 2 liter bleaching.jpg](https://commons.wikimedia.org/wiki/File:Coca-Cola_2_liter_bleaching.jpg) | CC0 |
