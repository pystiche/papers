import torch

from pystiche import data
from pystiche.image import transforms

__all__ = ["images"]


def image_note(url: str, mirror: bool = False) -> str:
    note = "The image is cropped"
    if mirror:
        note += " and mirrored"
    return f"{note}. The unprocessed image can be downloaded from {url}"


def make_image_transform(image: str) -> transforms.ComposedTransform:
    image_transform: transforms.Transform
    if image == "emma":
        image_transform = transforms.Crop(origin=(30, 12), size=(930, 682))
    elif image == "jenny":

        class MirrorHorizontally(transforms.Transform):
            def forward(self, image: torch.Tensor) -> torch.Tensor:
                return image.flip(2)

        image_transform = transforms.ComposedTransform(
            transforms.Crop(origin=(211, 462), size=(1843, 1386)), MirrorHorizontally()
        )
    elif image == "s":
        image_transform = transforms.Crop(origin=(159, 486), size=(2157, 1642))
    else:
        raise RuntimeError

    return transforms.ComposedTransform(
        image_transform,
        transforms.Resize(384, edge="vert", interpolation_mode="bicubic"),
    )


def images() -> data.DownloadableImageCollection:
    images_ = {
        "emma": data.DownloadableImage(
            "https://live.staticflickr.com/1/2281680_656225393e_o_d.jpg",
            author="monsieuricon (mricon)",
            title="Emma",
            date="17.12.2004",
            transform=make_image_transform("emma"),
            license=data.CreativeCommonsLicense(("by", "sa"), "2.0"),
            note=image_note("https://www.flickr.com/photos/mricon/2281680/"),
            md5="e3befabfd0215357e580b07e7d0ed05a",
        ),
        "jenny": data.DownloadableImage(
            "https://live.staticflickr.com/8626/16426686859_f882b3d317_o_d.jpg",
            author="Vidar Schiefloe (lydhode)",
            title="Jenny",
            date="06.02.2015",
            license=data.CreativeCommonsLicense(("by", "sa"), "2.0"),
            transform=make_image_transform("jenny"),
            note=image_note(
                "https://www.flickr.com/photos/lydhode/16426686859/", mirror=True,
            ),
            md5="387eeb2d8cd1bf19d14c263e078bb0a1",
        ),
        "blue_bottle": data.DownloadableImage(
            "https://raw.githubusercontent.com/chuanli11/CNNMRF/master/data/content/potrait1.jpg",
            title="Blue Bottle",
            author="Christopher Michel (cmichel67)",
            date="02.09.2014",
            license=data.NoLicense(),
            note=image_note("https://www.flickr.com/photos/cmichel67/15112861945"),
            md5="cb29d11ef6e1be7e074aa58700110e4f",
        ),
        "self-portrait": data.DownloadableImage(
            "https://raw.githubusercontent.com/chuanli11/CNNMRF/master/data/style/picasso.jpg",
            title="Self-Portrait",
            author="Pablo Ruiz Picasso",
            date="1907",
            license=data.ExpiredCopyrightLicense(1973),
            note=image_note("https://www.pablo-ruiz-picasso.net/images/works/57.jpg"),
            md5="4bd9c963fd52feaa940083f07e259aea",
        ),
        "s": data.DownloadableImage(
            "https://live.staticflickr.com/7409/9270411440_cdc2ee9c35_o_d.jpg",
            author="theilr",
            title="S",
            date="18.09.2011",
            license=data.CreativeCommonsLicense(("by", "sa"), "2.0"),
            transform=make_image_transform("s"),
            note=image_note("https://www.flickr.com/photos/theilr/9270411440/"),
            md5="525550983f7fd36d3ec10fba735ad1ef",
        ),
        "composition_viii": data.DownloadableImage(
            "https://www.wassilykandinsky.net/images/works/50.jpg",
            title="Composition VIII",
            author="Wassily Kandinsky",
            date="1923",
            license=data.ExpiredCopyrightLicense(1944),
            md5="c39077aaa181fd40d7f2cd00c9c09619",
        ),
    }
    return data.DownloadableImageCollection(images_)
