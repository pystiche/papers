from pystiche import data

__all__ = ["images"]


def images() -> data.DownloadableImageCollection:
    images_ = {
        "house": data.DownloadableImage(
            "https://associateddesigns.com/sites/default/files/plan_images/main/craftsman_house_plan_tillamook_30-519-picart.jpg",
            title="House Concept Tillamook",
            date="2014",
            license=data.NoLicense(),
            md5="5629bf7b24a7c98db2580ec2a8d784e9",
            guides=data.DownloadableImageCollection(
                {
                    "building": data.DownloadableImage(
                        "https://download.pystiche.org/images/house/building.png",
                        file="building.png",
                        author="Philip Meier",
                        license=data.PublicDomainLicense(),
                        md5="af6f7a11bdfb674cded10aa0c83021ad",
                    ),
                    "sky": data.DownloadableImage(
                        "https://download.pystiche.org/images/house/sky.png",
                        file="sky.png",
                        author="Philip Meier",
                        license=data.PublicDomainLicense(),
                        md5="5070c703dae2cf62009e7d64ba839b1b",
                    ),
                }
            ),
        ),
        "watertown": data.DownloadableImage(
            "https://ae01.alicdn.com/img/pb/136/085/095/1095085136_084.jpg",
            title="Watertown",
            author="Shop602835 Store",
            license=data.NoLicense(),
            md5="4cc98a503da5ce6eab0649b09fd3cf77",
            guides=data.DownloadableImageCollection(
                {
                    "building": data.DownloadableImage(
                        "https://download.pystiche.org/images/watertown/building.png",
                        file="building.png",
                        author="Philip Meier",
                        license=data.PublicDomainLicense(),
                        md5="cca6f4f2877dec1ea422fc23c644b672",
                    ),
                    "sky": data.DownloadableImage(
                        "https://download.pystiche.org/images/watertown/sky.png",
                        file="sky.png",
                        author="Philip Meier",
                        license=data.PublicDomainLicense(),
                        md5="4fdfc4ac4479dff3b6a1cf71d724f758",
                    ),
                }
            ),
        ),
        "wheat_field": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg/1920px-Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg",
            title="Wheat Field with Cypresses",
            author="Vincent van Gogh",
            date="1889",
            license=data.ExpiredCopyrightLicense(1890),
            md5="bfd085d7e800459c8ffb44fa404f73c3",
            guides=data.DownloadableImageCollection(
                {
                    "foreground": data.DownloadableImage(
                        "https://download.pystiche.org/images/wheat_field/foreground.png",
                        file="foreground.png",
                        author="Philip Meier",
                        license=data.PublicDomainLicense(),
                        md5="696d73f7a5f4ea0b436ac1bbfb57cb37",
                    ),
                    "sky": data.DownloadableImage(
                        "https://download.pystiche.org/images/wheat_field/sky.png",
                        file="sky.png",
                        author="Philip Meier",
                        license=data.PublicDomainLicense(),
                        md5="1102f4c57033c48be7d2142c8d6a3a29",
                    ),
                }
            ),
        ),
        "schultenhof": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/8/82/Schultenhof_Mettingen_Bauerngarten_8.jpg",
            title="Schultenhof Mettingen Bauerngarten 8",
            author="J.-H. Janßen",
            date="July 2010",
            license=data.CreativeCommonsLicense(("by", "sa"), "3.0"),
            md5="23f75f148b7b94d932e599bf0c5e4c8e",
        ),
        "starry_night": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/9/94/Starry_Night_Over_the_Rhone.jpg",
            title="Starry Night Over the Rhone",
            author="Vincent Willem van Gogh",
            date="1888",
            license=data.ExpiredCopyrightLicense(1890),
            md5="406681ec165fa55c26cb6f988907fe11",
        ),
    }
    return data.DownloadableImageCollection(images_)
