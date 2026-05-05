import { create } from 'zustand'

interface FilterStore {
  module: string
  tiers: string[]
  setModule: (m: string) => void
  setTiers: (t: string[]) => void
}

export const useFilterStore = create<FilterStore>(set => ({
  module: 'All',
  tiers: ['High', 'Medium', 'Low'],
  setModule: module => set({ module }),
  setTiers: tiers => set({ tiers }),
}))
