from pystiche.data import (
    CreativeCommonsLicense,
    DownloadableImage,
    DownloadableImageCollection,
    ExpiredCopyrightLicense,
    NoLicense,
)
from pystiche.image.transforms import ComposedTransform, Crop, Resize

__all__ = ["li_wand_2016_images"]


def image_note(url: str, mirror: bool = False) -> str:
    note = "The image is cropped"
    if mirror:
        note += " and mirrored"
    return f"{note}. The unprocessed image can be downloaded from {url}"


def transforms(image: str) -> ComposedTransform:
    origins_and_sizes = {
        "emma": ((30, 12), (930, 682)),
        "jenny": ((211, 462), (1843, 1386)),
        "s": ((159, 486), (1642, 2157)),
    }

    try:
        origin, size = origins_and_sizes[image]
    except KeyError:
        # TODO: add message
        raise RuntimeError

    return ComposedTransform(
        Crop(origin, size), Resize(384, edge="vert", interpolation_mode="bicubic")
    )


def li_wand_2016_images() -> DownloadableImageCollection:
    images = {
        "emma": DownloadableImage(
            "https://live.staticflickr.com/1/2281680_656225393e_o_d.jpg",
            author="monsieuricon (mricon)",
            title="Emma",
            date="17.12.2004",
            transform=transforms("emma"),
            license=CreativeCommonsLicense(("by", "sa"), "2.0"),
            note=image_note("https://www.flickr.com/photos/mricon/2281680/"),
            md5="7a10a2479864f394b4f06893b9202915",
        ),
        "jenny": DownloadableImage(
            "https://live.staticflickr.com/8626/16426686859_f882b3d317_o_d.jpg",
            author="Vidar Schiefloe (lydhode)",
            title="Jenny",
            date="06.02.2015",
            license=CreativeCommonsLicense(("by", "sa"), "2.0"),
            transform=transforms("jenny"),
            note=image_note(
                "https://www.flickr.com/photos/lydhode/16426686859/", mirror=True,
            ),
            md5="5b3442909ff850551c9baea433319508",
        ),
        "blue_bottle": DownloadableImage(
            "https://raw.githubusercontent.com/chuanli11/CNNMRF/master/data/content/potrait1.jpg",
            title="Blue Bottle",
            author="Christopher Michel (cmichel67)",
            date="02.09.2014",
            license=NoLicense(),
            note=image_note("https://www.flickr.com/photos/cmichel67/15112861945"),
            md5="cb29d11ef6e1be7e074aa58700110e4f",
        ),
        "self-portrait": DownloadableImage(
            "https://raw.githubusercontent.com/chuanli11/CNNMRF/master/data/style/picasso.jpg",
            title="Self-Portrait",
            author="Pablo Ruiz Picasso",
            date="1907",
            license=ExpiredCopyrightLicense(1973),
            note=image_note("https://www.pablo-ruiz-picasso.net/images/works/57.jpg"),
            md5="4bd9c963fd52feaa940083f07e259aea",
        ),
        "s": DownloadableImage(
            "https://live.staticflickr.com/7409/9270411440_cdc2ee9c35_o_d.jpg",
            author="theilr",
            title="S",
            date="18.09.2011",
            license=CreativeCommonsLicense(("by", "sa"), "2.0"),
            transform=transforms("s"),
            note=image_note("https://www.flickr.com/photos/theilr/9270411440/"),
            md5="5d78432b5ca703bb85647274a5e41656",
        ),
        "composition_viii": DownloadableImage(
            "https://www.wassilykandinsky.net/images/works/50.jpg",
            title="Composition VIII",
            author="Wassily Kandinsky",
            date="1923",
            license=ExpiredCopyrightLicense(1944),
            md5="c39077aaa181fd40d7f2cd00c9c09619",
        ),
    }
    return DownloadableImageCollection(images)
