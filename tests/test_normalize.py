import pytest

from product_matching.attributes import (
    conflicts,
    extract_codes,
    extract_colors,
    extract_specs,
    is_code,
)
from product_matching.normalize import fix_encoding, normalize_text, tokenize


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Notebook Sony 4 GB", "notebook sony 4gb"),
        ("notebook sony 4gb", "notebook sony 4gb"),
        ("NOTEBOOK SONY 4GB", "notebook sony 4gb"),
        ("Notebook Sony 4096 MB", "notebook sony 4gb"),
        ("Câmera Digital", "camera digital"),
        ("Colchão Lençol Ação", "colchao lencol acao"),
        ("18.000 BTUs", "18000btus"),
        ("2,4 GHz", "2.4ghz"),
        ('TV LCD 32"', "tv lcd 32pol"),
        ("TV 32 Polegadas", "tv 32pol"),
        ("306 Litros", "306l"),
        ("Ar Split MSS-18CR", "ar split mss18cr"),
        ("Roteador Wi-Fi", "roteador wi fi"),
        ("Tapete 50 x 70 cm", "tapete 50x70cm"),
        ("Câmera 14.2 Megapixels", "camera 14.2mp"),
        ("Furadeira 220 V", "furadeira 220v"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_mb_not_converted_when_not_a_multiple_of_1024():
    assert normalize_text("Pen drive 500 MB") == "pen drive 500mb"


def test_fix_encoding_repairs_mojibake():
    assert fix_encoding("Notebook Samsung SÃ©rie 9") == "Notebook Samsung Série 9"
    assert normalize_text("IntelÂ® Core i5") == "intel core i5"


def test_tokenize_drops_stopwords():
    assert tokenize(normalize_text("Kit de Panelas c/ Tampa para Fogão")) == [
        "kit",
        "panelas",
        "tampa",
        "fogao",
    ]


def test_query_and_listing_agree_on_units():
    assert "4gb" in normalize_text("notebook sony 4 GB").split()
    assert "4gb" in normalize_text("Notebook VAIO Intel Core i3 4096 MB 500 GB").split()


@pytest.mark.parametrize(
    "tok, expected",
    [
        ("un40c6900", True),
        ("mss18cr", True),
        ("66906760", True),
        ("4gb", False),
        ("220v", False),
        ("50x70cm", False),
        ("2012", False),
        ("tv", False),
        ("i5", False),
    ],
)
def test_is_code(tok, expected):
    assert is_code(tok) is expected


def test_extract_codes():
    name = normalize_text("Samsung UN40C6900 LED Plana 40 Polegadas 4 GB")
    assert extract_codes(name) == ["un40c6900"]


def test_extract_colors():
    assert extract_colors(normalize_text("Celular Preto/Vermelho")) == ["Preto", "Vermelho"]
    assert extract_colors(normalize_text("Tanque Aço Escovado")) == ["Aço Escovado"]
    assert extract_colors(normalize_text("Notebook Sony")) == []


def test_specs_and_conflicts():
    a = extract_specs(normalize_text("Logic FL738 8GB"))
    b = extract_specs(normalize_text("Logic FL738 4GB"))
    c = extract_specs(normalize_text("Logic FL738"))
    assert conflicts(a["capacity"], b["capacity"])
    assert not conflicts(a["capacity"], c["capacity"])  # unstated is not a conflict
    assert extract_specs("hd 1tb")["capacity"] == frozenset({"1024gb"})
