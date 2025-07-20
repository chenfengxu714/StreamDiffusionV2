<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  import Button from './Button.svelte';
  import { promptValues, sendPromptValues } from '$lib/store';
  
  export let params: FieldProps;
  let promptValue: string = '';
  
  onMount(() => {
    promptValue = String(params?.default ?? '');
    // Initialize promptValues store with the default value
    $promptValues[params.id] = promptValue;
  });
  
  function handleSendPrompt() {
    // Update the promptValues store with current value
    $promptValues[params.id] = promptValue;
    // Send the prompt values to the pipeline
    sendPromptValues();
  }
  
  // Watch for changes in promptValue and update the store
  $: if (promptValue !== undefined) {
    $promptValues[params.id] = promptValue;
  }
</script>

<div class="flex flex-col gap-3">
  <label class="text-sm font-medium" for={params?.title}>
    {params?.title}
  </label>
  <div class="flex gap-2">
    <div class="flex-1">
      <textarea
        class="w-full px-3 py-2 font-light outline-none rounded-md border border-gray-700 dark:text-black"
        title={params?.title}
        placeholder="Add your prompt here..."
        bind:value={promptValue}
      ></textarea>
    </div>
    <Button 
      on:click={handleSendPrompt}
      classList={'px-4 py-2 self-end'}
      disabled={!promptValue.trim()}
    >
      Send
    </Button>
  </div>
</div> 