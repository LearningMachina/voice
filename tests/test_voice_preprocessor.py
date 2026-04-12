"""Tests for the voice preprocessor module."""

import pytest

from text_processing.voice_preprocessor import clean_for_speech


class TestFencedCodeBlocks:
    def test_fenced_code_block_replaced(self):
        text = "Here is an example:\n```\nprint('hello')\n```\nNeat, right?"
        result = clean_for_speech(text)
        assert "```" not in result
        assert "See the example in the chat." in result

    def test_fenced_code_block_with_language(self):
        text = "```python\nx = 1\n```"
        assert clean_for_speech(text) == "See the example in the chat."


class TestInlineCode:
    def test_simple_identifier_kept(self):
        assert clean_for_speech("Use the `request` object") == "Use the request object"

    def test_complex_expression_verbalized(self):
        result = clean_for_speech("Run `obj.method()` to start")
        assert "obj dot method" in result
        assert "`" not in result

    def test_operator_expression_cleaned(self):
        result = clean_for_speech("Check `a + b` for the sum")
        # Operators stripped, identifiers extracted or omitted — no "see the code"
        assert "see the code in the chat" not in result
        assert "`" not in result

    def test_underscore_identifier_kept(self):
        assert clean_for_speech("Set `my_var` here") == "Set my_var here"


class TestBoldItalic:
    def test_bold_stripped(self):
        assert clean_for_speech("Hello **world**") == "Hello world"

    def test_italic_stripped(self):
        assert clean_for_speech("This is *important*") == "This is important"

    def test_bold_and_italic(self):
        assert clean_for_speech("**bold** and *italic*") == "bold and italic"


class TestNumberedList:
    def test_first_three(self):
        text = "1. Open the file\n2. Edit it\n3. Save"
        result = clean_for_speech(text)
        assert result.startswith("First, Open the file")
        assert "Second, Edit it" in result
        assert "Third, Save" in result

    def test_beyond_ten(self):
        text = "11. Do something"
        assert clean_for_speech(text) == "Step 11, Do something"

    def test_tenth(self):
        text = "10. Last ordinal"
        assert clean_for_speech(text) == "Tenth, Last ordinal"


class TestBulletMarkers:
    def test_dash_bullet_stripped(self):
        assert clean_for_speech("- Buy milk") == "Buy milk"

    def test_asterisk_bullet_stripped(self):
        assert clean_for_speech("* Buy eggs") == "Buy eggs"


class TestURLs:
    def test_url_replaced(self):
        text = "Visit https://example.com for more"
        result = clean_for_speech(text)
        assert "https://" not in result
        assert "see the link in the chat" in result

    def test_http_url_replaced(self):
        text = "Go to http://example.com/path?q=1"
        result = clean_for_speech(text)
        assert "http://" not in result
        assert "see the link in the chat" in result


class TestAbbreviations:
    def test_eg(self):
        assert "for example" in clean_for_speech("Use a library, e.g. NumPy")

    def test_ie(self):
        assert "that is" in clean_for_speech("The default, i.e. none")

    def test_etc(self):
        assert "and so on" in clean_for_speech("lists, dicts, etc.")

    def test_vs(self):
        assert "versus" in clean_for_speech("Python vs. Ruby")


class TestWhitespace:
    def test_multiple_spaces_collapsed(self):
        assert clean_for_speech("hello   world") == "hello world"

    def test_leading_trailing_stripped(self):
        assert clean_for_speech("  hello  ") == "hello"

    def test_whitespace_only_input(self):
        assert clean_for_speech("   ") == ""


class TestPassthrough:
    def test_empty_string(self):
        assert clean_for_speech("") == ""

    def test_already_clean(self):
        assert clean_for_speech("Hello world") == "Hello world"


class TestCombined:
    def test_mixed_markdown(self):
        text = "1. Use **bold** and visit https://x.com"
        result = clean_for_speech(text)
        assert result == "First, Use bold and visit see the link in the chat"

    def test_inline_code_with_abbreviation(self):
        text = "Call `my_func` e.g. in a loop"
        result = clean_for_speech(text)
        assert "my_func" in result
        assert "for example" in result
        assert "`" not in result


class TestVerbalizeCode:
    def test_short_code_with_dot_and_equals(self):
        result = clean_for_speech('Check `key.code == "up"` now')
        assert "key dot code equals up" in result

    def test_arrow_function(self):
        result = clean_for_speech("Use `a => b` syntax")
        assert "a arrow b" in result

    def test_not_equals(self):
        result = clean_for_speech("Check `x !== y` here")
        assert "x not equals y" in result

    def test_arithmetic_operators(self):
        assert "plus" in clean_for_speech("the `+` operator")
        assert "minus" in clean_for_speech("the `-` operator")
        assert "times" in clean_for_speech("the `*` operator")

    def test_arithmetic_expression(self):
        result = clean_for_speech("compute `a + b`")
        assert "a plus b" in result

    def test_long_code_extracts_identifiers(self):
        # Long code with operators — identifiers are extracted
        long_code = "x = " + "func(a, b, c) + " * 5 + "end"
        result = clean_for_speech(f"Run `{long_code}` now")
        # The code is too complex to verbalize but identifiers can be extracted
        # or it gets omitted entirely — either way, no "see the code in the chat"
        assert "see the code in the chat" not in result

    def test_code_with_slash_extracts_identifiers(self):
        result = clean_for_speech("Use `app.get('/')` here")
        assert "app get" in result
        assert "see the code in the chat" not in result

    def test_rust_style_code_extracts_identifiers(self):
        result = clean_for_speech(
            "generates a random number using `rand::thread_rng().gen_range(1..100)`."
        )
        assert "rand" in result
        assert "thread rng" in result
        assert "gen range" in result
        assert "::" not in result


class TestStreamingVoicePreprocessor:
    def test_basic_text_passes_through(self):
        from text_processing.voice_preprocessor import StreamingVoicePreprocessor
        p = StreamingVoicePreprocessor()
        assert p.clean("Hello world") == "Hello world"

    def test_code_block_across_chunks_suppressed(self):
        from text_processing.voice_preprocessor import StreamingVoicePreprocessor
        p = StreamingVoicePreprocessor()
        r1 = p.clean("Here is code:\n```python")
        r2 = p.clean("x = 1\ny = 2")
        r3 = p.clean("```\nAfter the code.")
        assert "See the example in the chat" in r1
        assert r2 == ""
        assert "After the code" in r3

    def test_announces_code_block_only_once(self):
        from text_processing.voice_preprocessor import StreamingVoicePreprocessor
        p = StreamingVoicePreprocessor()
        p.clean("```python")
        p.clean("some code")
        p.clean("```")
        r = p.clean("```js")
        # Second code block should not re-announce
        assert r == ""

    def test_disabled_passes_through(self):
        from text_processing.voice_preprocessor import StreamingVoicePreprocessor
        p = StreamingVoicePreprocessor(enabled=False)
        assert p.clean("```python\nx = 1\n```") == "```python\nx = 1\n```"

    def test_inline_code_still_cleaned(self):
        from text_processing.voice_preprocessor import StreamingVoicePreprocessor
        p = StreamingVoicePreprocessor()
        result = p.clean("Use `express` for this.")
        assert "express" in result
        assert "`" not in result

    def test_multiple_code_blocks_suppressed(self):
        from text_processing.voice_preprocessor import StreamingVoicePreprocessor
        p = StreamingVoicePreprocessor()
        r1 = p.clean("First:\n```\ncode1\n```")
        r2 = p.clean("Second:\n```\ncode2\n```")
        assert "See the example" in r1
        assert "Second" in r2
        assert "code2" not in r2
