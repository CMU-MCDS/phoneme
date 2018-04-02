from typing import Any, Optional

class Tag:
    key = ...  # type: Any
    value = ...  # type: Any
    def __init__(self, key: Optional[Any] = ..., value: Optional[Any] = ...) -> None: ...
    def startElement(self, name, attrs, connection): ...
    def endElement(self, name, value, connection): ...
    def to_xml(self): ...
    def __eq__(self, other): ...

class TagSet(list):
    def startElement(self, name, attrs, connection): ...
    def endElement(self, name, value, connection): ...
    def add_tag(self, key, value): ...
    def to_xml(self): ...

class Tags(list):
    def startElement(self, name, attrs, connection): ...
    def endElement(self, name, value, connection): ...
    def to_xml(self): ...
    def add_tag_set(self, tag_set): ...
