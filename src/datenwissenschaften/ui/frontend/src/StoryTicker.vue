<script setup>
defineProps({ events: { type: Array, required: true } })

const ICONS = { milestone: '★', good: '▲', bad: '▼' }
</script>

<template>
  <TransitionGroup name="ticker" tag="ul" class="story-ticker">
    <li v-for="item in events" :key="item.key" :class="['story-event', item.kind]">
      <b>{{ ICONS[item.kind] }}</b>
      <strong>{{ item.text }}</strong>
      <span v-if="item.detail">{{ item.detail }}</span>
    </li>
  </TransitionGroup>
</template>
