<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  import Button from './Button.svelte';
  import { promptValues, sendPromptValues } from '$lib/store';
  export let value: string;
  export let params: FieldProps;
  let showUpdatedNotice = false;
  let hideNoticeTimeout: number | null = null;
  onMount(() => {
    value = String(params?.default ?? '');
    // keep prompt store in sync for video mode too
    $promptValues[params.id] = value;
  });

  function handleSendPrompt() {
    $promptValues[params.id] = value;
    sendPromptValues();
    showUpdatedNotice = true;
    if (hideNoticeTimeout) {
      clearTimeout(hideNoticeTimeout);
    }
    hideNoticeTimeout = window.setTimeout(() => {
      showUpdatedNotice = false;
      hideNoticeTimeout = null;
    }, 2000);
  }
</script>

<div class="flex flex-col gap-3 relative">
  <label class="text-sm font-medium" for={params?.title}>
    {params?.title}
  </label>
  <div class="text-normal flex items-center rounded-md border border-gray-700">
    <textarea
      class="mx-1 w-full px-3 py-2 font-light outline-none dark:text-black"
      title={params?.title}
      placeholder="Add your prompt here..."
      bind:value
    ></textarea>
  </div>
  {#if showUpdatedNotice}
    <div class="mt-1 inline-block w-max rounded-md bg-green-100 px-2 py-1 text-sm text-green-800 dark:bg-green-900 dark:text-green-200">Prompt updated</div>
  {/if}
  <div class="flex justify-center mt-2 w-full">
    <Button on:click={handleSendPrompt} classList={'px-4 py-2'} disabled={!value?.trim?.()}>Send</Button>
  </div>
</div>
