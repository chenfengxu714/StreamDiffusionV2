<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  import Button from './Button.svelte';
  import { promptValues, sendPromptValues } from '$lib/store';
  
  export let params: FieldProps;
  let promptValue: string = '';
  let showUpdatedNotice = false;
  let hideNoticeTimeout: number | null = null;
  
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
    // Show transient confirmation
    showUpdatedNotice = true;
    if (hideNoticeTimeout) {
      clearTimeout(hideNoticeTimeout);
    }
    hideNoticeTimeout = window.setTimeout(() => {
      showUpdatedNotice = false;
      hideNoticeTimeout = null;
    }, 2000);
  }
  
  // Watch for changes in promptValue and update the store
  $: if (promptValue !== undefined) {
    $promptValues[params.id] = promptValue;
  }
</script>

<div class="flex flex-col gap-3 relative">
  <label class="text-sm font-medium" for={params?.title}>
    {params?.title}
  </label>
  <div class="flex-1">
    <textarea
      class="w-full px-3 py-2 font-light outline-none rounded-md border border-gray-700 dark:text-black"
      title={params?.title}
      placeholder="Add your prompt here..."
      bind:value={promptValue}
    ></textarea>
  </div>
  {#if showUpdatedNotice}
    <div class="mt-1 inline-block w-max rounded-md bg-green-100 px-2 py-1 text-sm text-green-800 dark:bg-green-900 dark:text-green-200">
      Prompt updated
    </div>
  {/if}
  <div class="flex justify-center mt-2 w-full">
    <Button 
      on:click={handleSendPrompt}
      classList={'px-4 py-2'}
      disabled={!promptValue.trim()}
    >
      Send
    </Button>
  </div>
</div> 