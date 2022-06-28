from typing import Optional

from pystiche import data

__all__ = ["images"]


def license(original: Optional[str] = None) -> str:
    license = (
        "The image is part of a repository that is published for academic and "
        "non-commercial use only "
        "(https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/README.md?plain=1#L48)."
    )
    if original:
        license = f"{license} The original was probably downloaded from {original}. "
    return f"{license} Proceed at your own risk."


def images() -> data.DownloadableImageCollection:
    return data.DownloadableImageCollection(
        {
            "house": data.DownloadableImage(
                "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_content.jpg?raw=true",
                file="house_concept_tillamook.jpg",
                license=license(
                    "https://associateddesigns.com/sites/default/files/plan_images/main/craftsman_house_plan_tillamook_30-519-picart.jpg"
                ),
                md5="5629bf7b24a7c98db2580ec2a8d784e9",
                guides=data.DownloadableImageCollection(
                    {
                        "building": data.DownloadableImage(
                            "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style1_nosky.jpg?raw=true",
                            file="building.jpg",
                            author="Gatys et al.",
                            license=license(),
                            md5="1fa945871244cf1cefc5e08f8da83fdf",
                        ),
                        "sky": data.DownloadableImage(
                            "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style1_sky.jpg?raw=true",
                            file="sky.jpg",
                            author="Gatys et al.",
                            license=license(),
                            md5="3c21f0d573cc73a6b58b9d559117805b",
                        ),
                    }
                ),
            ),
            "watertown": data.DownloadableImage(
                "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style1.jpg?raw=true",
                file="watertown.jpg",
                license=license("https://de.aliexpress.com/item/1705231003.html"),
                md5="4cc98a503da5ce6eab0649b09fd3cf77",
                guides=data.DownloadableImageCollection(
                    {
                        "building": data.DownloadableImage(
                            "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style1_nosky.jpg?raw=true",
                            file="building.jpg",
                            author="Gatys et al.",
                            license=license(),
                            md5="1fa945871244cf1cefc5e08f8da83fdf",
                        ),
                        "sky": data.DownloadableImage(
                            "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style1_sky.jpg?raw=true",
                            file="sky.jpg",
                            author="Gatys et al.",
                            license=license(),
                            md5="3c21f0d573cc73a6b58b9d559117805b",
                        ),
                    }
                ),
            ),
            "wheat_field": data.DownloadableImage(
                "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style2.jpg?raw=true",
                file="wheat_field.jpg",
                license=license(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg/1920px-Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg"
                ),
                md5="4af9e64534c055bf7db5ee3ab7daf608",
                guides=data.DownloadableImageCollection(
                    {
                        "foreground": data.DownloadableImage(
                            "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style2_nosky.jpg?raw=true",
                            file="foreground.jpg",
                            author="Gatys et al.",
                            license=license(),
                            md5="67c6e653f4aa629140cb2fc53a3406d9",
                        ),
                        "sky": data.DownloadableImage(
                            "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig2_style2_nosky.jpg?raw=true",
                            file="sky.jpg",
                            author="Gatys et al.",
                            license=license(),
                            md5="67c6e653f4aa629140cb2fc53a3406d9",
                        ),
                    }
                ),
            ),
            "schultenhof": data.DownloadableImage(
                "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig3_content.jpg?raw=true",
                file="schultenhof.jpg",
                license=license(
                    "https://upload.wikimedia.org/wikipedia/commons/8/82/Schultenhof_Mettingen_Bauerngarten_8.jpg"
                ),
                md5="23f75f148b7b94d932e599bf0c5e4c8e",
            ),
            "starry_night": data.DownloadableImage(
                "https://github.com/leongatys/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/Images/ControlPaper/fig3_style1.jpg?raw=true",
                file="starry_night_over_the_rhone.jpg",
                license=license(
                    "https://upload.wikimedia.org/wikipedia/commons/9/94/Starry_Night_Over_the_Rhone.jpg"
                ),
                md5="e67c25e4aa6070cc4e5ab7f3ce91c218",
            ),
        }
    )
