# Understanding the Default Plugin

This tutorial explains the role and behavior of the `DefaultPlugin` in the aclarai import system. Unlike other plugins that target specific formats, the `DefaultPlugin` is a general-purpose fallback that ensures any file can be processed.

## Key Concepts

-   **Fallback Mechanism**: The `DefaultPlugin` has a low priority and is designed to handle files that are not claimed by any other, more specific plugin.
-   **Automatic Discovery**: You do **not** need to manually register or enable the `DefaultPlugin`. The `PluginManager` automatically discovers and includes it in the import process.
-   **Best-Effort Conversion**: It uses a combination of pattern matching and (optional) LLM-based analysis to extract conversations from unstructured text.

## How It Works

When the `ImportOrchestrator` processes a file, it asks the `PluginManager` for the best plugin. The `PluginManager` checks all high-priority plugins first. If none of them can handle the file, the `DefaultPlugin` is chosen as the fallback.

Its `can_accept()` method always returns `True`, guaranteeing that it will accept any file that other plugins have rejected.

## Usage

You don't need to do anything to "use" the `DefaultPlugin`. It works automatically as part of the `Tier1ImportSystem`. When you import a file that doesn't match any specific format (like a plain `.txt` file with a simple `speaker: message` structure), the `DefaultPlugin` will be used behind the scenes.

```python
from pathlib import Path
from aclarai_shared.config import aclaraiConfig
from aclarai_shared.import_system import Tier1ImportSystem, ImportStatus
from aclarai_shared.plugin_manager import PluginManager

# Standard import system setup
config = aclaraiConfig()
plugin_manager = PluginManager()
system = Tier1ImportSystem(config=config, plugin_manager=plugin_manager)

# Create a generic text file that no specific plugin would claim
generic_file = Path("generic_chat.txt")
generic_file.write_text("alex: This is a generic chat.\njane: It should be handled by the default plugin.")

# Process the file
result = system.import_file(generic_file)

# Check the result
if result.status == ImportStatus.SUCCESS:
    # You can inspect the plugin metadata to confirm the DefaultPlugin was used
    markdown_content = result.output_file.read_text()
    if 'plugin_metadata={"source_format": "fallback_llm"}' in markdown_content:
        print("Confirmed: The DefaultPlugin was used as a fallback.")

# Clean up
generic_file.unlink()
if result.output_file and result.output_file.exists():
    result.output_file.unlink()
```

## Supported Formats

The `DefaultPlugin` is designed to recognize common, simple formats out-of-the-box:

-   **Simple Speaker Format**: `speaker: message`
-   **Log-style Format**: `ENTRY [timestamp] speaker >> message`
-   **Metadata Headers**: It can often extract metadata like `TOPIC:` or `PARTICIPANTS:` from the top of a file.

If these patterns fail, it can optionally use a configured LLM to attempt a more intelligent extraction.

## Customizing LLM Prompts

For advanced use cases, you can customize the LLM prompts that the `DefaultPlugin` uses for conversation extraction. This allows you to tailor its behavior to specific types of unstructured text you frequently work with.

Prompt templates are located in `settings/prompts/`. By editing the `conversation_extraction.yaml` file, you can fine-tune the instructions given to the LLM, improving its accuracy for your data.

For more details, see the [Plugin System Guide (`docs/guides/plugin_system_guide.md`)](<../guides/plugin_system_guide.md>).
