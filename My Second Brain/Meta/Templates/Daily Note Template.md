---
category: 
tags:
  - daily
created: 2023-12-14T11:54
updated: 2023-12-27T11:42
---
## Notes
- 

## Yesterday's Recap
```dataview  
list 
where file.cday= date(yesterday)  
```



>[!EXAMPLE]- Files Added and Modified Today
>### New Today 
>```dataview
>LIST 
>FROM ""
>WHERE file.cday = this.file.cday 
>AND file.path != this.file.path
>SORT file.name asc
>```
>
>### Modified Today 
>```dataview
LIST
FROM ""
WHERE file.path != this.file.path 
AND
file.mday = this.file.mday
>```

