import { createApp } from 'vue'
import App from './App.vue'
import StreamView from './StreamView.vue'
import './style.css'

createApp(window.location.pathname === '/stream' ? StreamView : App).mount('#app')
